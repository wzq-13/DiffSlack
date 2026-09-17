import numpy as np
import pickle
import time
import os 
import shutil
import contextlib
import io
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from typing import Tuple, Callable, Optional, Dict
import tqdm
import globalvar
from utils.utils import visualize_data_batch, check_polygon_intersection, get_rect_points_vectorized, visualize_data_batch_paper, path_smoothness,visualize_data_batch_paper2
from utils.prob import _create_objective_function, obj_fn, xy2xy_heading, soft_constraints, xy2xy_heading
from models.utils import create_model, path_clean
from torch.utils.tensorboard import SummaryWriter
from models.DiffSlack import AdaNP, AdaNPTest

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# DEVICE = torch.device("cpu")


def _synchronize_device():
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize(DEVICE)

class DiffSlack_Trainer:
    def __init__(self, config, train_dataset, val_dataset, test_dataset=None, save_dir=None, load_dir=None, log_dir=None):
        """Initializes the Trainer with data, method, and configuration."""
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True,num_workers=8,persistent_workers=True)
        self.val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True,num_workers=8,persistent_workers=False)
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,pin_memory=True,num_workers=2,persistent_workers=False)
        
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.adanp = AdaNP(
            n_outputs=80,
            n_constraints=200,
            max_depth=self.config['max_depth'],
            tol=self.config['inference_tol'],
            initialization_mode=self.config.get('slack_initialization', 'learned'),
            initialization_constant=self.config.get('slack_initialization_constant', 0.1),
            w_traj=self.config.get('w_traj', 5.0),
            w_slack=self.config.get('w_slack', 1.0),
            damping=self.config.get('damping', 1e-4),
        )
        self.adanp_test = AdaNPTest(
            max_depth=self.config['max_depth'],
            tol=self.config['inference_tol'],
            w_traj=self.config.get('w_traj', 5.0),
            w_slack=self.config.get('w_slack', 1.0),
            damping=self.config.get('damping', 1e-4),
        )
        self.training_tol = self.config['training_tol']
        self.constraint_func_stage1 = _create_objective_function(stage=1)
        self.constraint_func_stage2 = _create_objective_function(stage=2)
        
        if load_dir is not None:
            checkpoint = torch.load(load_dir, map_location=DEVICE)
            load_config = checkpoint.get('config', None)
            self.config['hidden_dim'] = load_config.get('hidden_dim', self.config['hidden_dim'])
            self.config['dropout'] = load_config.get('dropout', self.config['dropout'])
        self.model = create_model(self.config, device=DEVICE)
        learning_rate = self.config['lr']
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=self.config['weight_decay'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config['lr_decay_step'], gamma=self.config['lr_decay'])
        
        if self.save_dir is not None:
            print(f'Creating save directory at {self.save_dir}')
            os.makedirs(self.save_dir, exist_ok=True)
        if load_dir is not None:
            checkpoint = torch.load(load_dir, map_location=DEVICE)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f'optimizer lr: {self.optimizer.param_groups[0]["lr"]}')
            print(f'Model loaded from {load_dir}')
            
        self.loss_func = nn.MSELoss()
        self.adanp._original_forward = self.adanp.forward
        self.adanp.forward = torch.compile(self.adanp.forward, mode='default')
        
    def train_epoch_stage1(self, train_loader: DataLoader, epoch: int):
        """Trains the model for one epoch."""
        epoch_metrics = {'total_loss': 0.0, 'loss_map': 0.0, 'loss_soft': 0.0, 'loss_slack': 0.0}
        self.model.train()
        bar = tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{self.config['num_epochs_stage1']}")
        for X_batch in bar:
            for key in X_batch:
                X_batch[key] = X_batch[key].to(DEVICE, non_blocking=True)
            self.optimizer.zero_grad()
            Y_pred = self.model(X_batch)
            loss_map = obj_fn(X_batch, Y_pred, config=self.config)
            Y_pred_ = Y_pred.view(Y_pred.size(0), -1, self.config['N_dim'])  # (B, N, 2)
            xy_pred = Y_pred_[:,:,:2]
            
            xy_heading = xy2xy_heading(xy_pred)  # (B, N, 3)
            
            loss_cons = soft_constraints(xy_heading, self.config['obs_constraints_weight'], X_batch['obstacles_vertices'])
            
            loss_slack = self.constraint_func_stage1(X_batch, Y_pred).abs().mean()
            
            loss = loss_map + loss_cons + loss_slack * self.config['slack_weight']
            
            loss.backward()
            self.optimizer.step()
            bar.set_postfix(
                loss=f"{loss.item():.4f}",
                loss_cons=f"{loss_cons.item():.4f}",
                loss_slack=f"{loss_slack.item():.4f}"
            )
            epoch_metrics['total_loss'] += loss.item()
            epoch_metrics['loss_map'] += loss_map.item()
            epoch_metrics['loss_soft'] += loss_cons.item()
            epoch_metrics['loss_slack'] += loss_slack.item()
            
        self.scheduler.step()
        
        num_batches = len(train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
            
        return epoch_metrics
    
    def train_epoch_stage2(self, train_loader: DataLoader, epoch: int):
        """Trains the model for one epoch."""
        epoch_metrics = {'total_loss': 0.0, 'loss_map_proj': 0.0, 'loss_soft': 0.0, 'loss_soft_proj': 0.0, 'loss_proj': 0.0}
        self.model.train()
        bar = tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{self.config['num_epochs_stage2']}")
        for X_batch in bar:
            for key in X_batch:
                X_batch[key] = X_batch[key].to(DEVICE, non_blocking=True)
            self.optimizer.zero_grad()
            Y_pred = self.model(X_batch)
            B = Y_pred.size(0)
            Y_proj, depth, actual_depth = self.adanp(X_batch, Y_pred, self.constraint_func_stage2)
            loss_map_proj = obj_fn(X_batch, Y_proj, config=self.config)
            
            Y_pred_ = Y_pred.view(B, -1, self.config['N_dim'])
            Y_proj_ = Y_proj.view(B, -1, self.config['N_dim'])
            
            xy_heading = xy2xy_heading(Y_pred_[:,:,:2])  # (B, N, 3)
            xy_heading_proj = xy2xy_heading(Y_proj_[:,:,:2])  # (B, N, 3)
            end_point = Y_proj_[:, -1, :2]
            end_point_pred = Y_pred_[:, -1, :2]
            target_point = X_batch['target'][:, :2]
            loss_end = self.loss_func(end_point, target_point)
            loss_end_pred = self.loss_func(end_point_pred, target_point)

            loss_cons = soft_constraints(xy_heading, self.config['obs_constraints_weight'], X_batch['obstacles_vertices'])
            loss_cons_proj = soft_constraints(xy_heading_proj, self.config['obs_constraints_weight'], X_batch['obstacles_vertices'])
            
            Y_initial = self.adanp.initialize_output(
                X_batch, Y_pred, self.constraint_func_stage2
            )
            loss_proj = torch.mean((Y_proj.detach() - Y_initial)**2)
            
            loss = loss_map_proj + loss_end + loss_end_pred*5 + loss_cons + loss_proj * self.config['proj_loss_weight']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            bar.set_postfix(
                loss=f"{loss.item():.4f}",
                loss_map_proj=f"{loss_map_proj.item():.4f}",
                loss_cons=f"{loss_cons.item():.4f}",
                loss_cons_proj=f"{loss_cons_proj.item():.4f}",
                loss_proj=f"{loss_proj.item():.4f}",
            )
            epoch_metrics['total_loss'] += loss.item()
            epoch_metrics['loss_map_proj'] += loss_map_proj.item()
            epoch_metrics['loss_soft'] += loss_cons.item()
            epoch_metrics['loss_soft_proj'] += loss_cons_proj.item()
            epoch_metrics['loss_proj'] += loss_proj.item()
        self.scheduler.step()
        
        num_batches = len(train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
            
        return epoch_metrics
    
    def train(self, begin_epoch: int = 0):
        """Main training loop."""
        # self.save_path_data(self.test_loader)
        # self.test(self.test_loader)
        # self.test_visualization(save_path=self.log_dir)
        # return
        self.writer = SummaryWriter(log_dir=self.log_dir) if self.log_dir is not None else None
        num_epochs_stage1 = self.config['num_epochs_stage1']
        if begin_epoch < self.config['num_epochs_stage1']:
            for epoch in range(begin_epoch, num_epochs_stage1):
                train_metrics = self.train_epoch_stage1(self.train_loader, epoch)
                # if epoch == 2:
                #     self.model.kan.refine(5)
                print(f'Epoch {epoch+1}/{num_epochs_stage1}: {train_metrics}')
                if self.writer is not None:
                    for key in train_metrics:
                        self.writer.add_scalar(f'Train/{key}', train_metrics[key], epoch)
                if (epoch + 1) % self.config['eval_step'] == 0:
                    val_metrics = self.evaluate_stage1(self.val_loader)
                    print(f'--- Validation Loss------')
                    print(val_metrics)
                    if self.writer is not None:
                        for key in val_metrics:
                            self.writer.add_scalar(f'Val/{key}', val_metrics[key], epoch)

                if (epoch + 1) % self.config['save_step'] == 0 or epoch == num_epochs_stage1 - 1:
                    self._save_model(epoch=epoch)
                    # self.test_visualization(save_path=self.log_dir)
            self.test(self.test_loader, test_hard=False)
            
        del self.train_loader, self.val_loader
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config['batch_size']//2, shuffle=True, pin_memory=True,num_workers=8,persistent_workers=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.config['batch_size']//2, shuffle=False, pin_memory=True,num_workers=8,persistent_workers=False)
        num_epochs_stage2 = self.config['num_epochs_stage2']
        self.config['save_step'] = max(1, self.config['save_step'] // 4)
        
        print(f"Starting Stage 2 training with {num_epochs_stage2} epochs.")
        
        begin_epoch = max(begin_epoch, num_epochs_stage1)
        for epoch in range(begin_epoch, num_epochs_stage1 + num_epochs_stage2):
            train_metrics = self.train_epoch_stage2(self.train_loader, epoch)
            print(f'Epoch {epoch+1}/{num_epochs_stage1 + num_epochs_stage2}: {train_metrics}')
            if self.writer is not None:
                for key in train_metrics:
                    self.writer.add_scalar(f'Train/{key}', train_metrics[key], epoch)
            if (epoch + 1) % self.config['eval_step'] == 0:
                val_metrics = self.evaluate_stage2(self.val_loader)
                print(f'--- Validation Loss------')
                print(val_metrics)
                if self.writer is not None:
                    for key in val_metrics:
                        self.writer.add_scalar(f'Val/{key}', val_metrics[key], epoch)

            if (epoch + 1) % self.config['save_step'] == 0 or epoch == num_epochs_stage1 + num_epochs_stage2 - 1:
                self._save_model(epoch=epoch)
                # self.test_visualization(save_path=self.log_dir)
            
            if (epoch + 1) % 10 == 0:
                time.sleep(20)
        
        # self.test_visualization(save_path=self.log_dir)
        self.test(self.test_loader, test_hard=True)
    

    def compute_score(self, X_batch: torch.Tensor, Y_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Computes score."""
        B = Y_pred.size(0)
        # compute length
        Y_pred = Y_pred.view(B, -1, self.config['N_dim'])  # (B, N, 7)
        Y_pred = Y_pred[:, :, :2]  # (B, N, 2)
        Y_cleaned = path_clean(Y_pred[0].detach().cpu().numpy(), X_batch['target'][0, :2].detach().cpu().numpy())  # (N_cleaned, 2)
        lengths = np.sum(np.linalg.norm(np.diff(Y_cleaned, axis=0), axis=1)).item()
        
        Y_pred_cpu = Y_pred.detach().cpu().numpy()
        diffs = np.diff(Y_pred_cpu, axis=1)                          # (B, N-1, 2)
        dists = np.linalg.norm(diffs, axis=2)                   # (B, N-1)
        threshold = 1.0
        violations = np.maximum(0, dists - threshold)           # (B, N-1)
        dist_violation = np.mean(violations)
        
        xy_heading = xy2xy_heading(Y_pred)[0,1:,:]  # (N, 3)
        rect_points = get_rect_points_vectorized(xy_heading, width=globalvar.vehicle_geometrics_.vehicle_width, length=globalvar.vehicle_geometrics_.vehicle_length)  # (N, 4, 2)
        obstacles = X_batch['obstacles_vertices'][0]  # (M, 4, 2)
        collision = False
        for i in range(rect_points.shape[0]):
            for j in range(obstacles.shape[0]):
                if check_polygon_intersection(rect_points[i].cpu().numpy(), obstacles[j].cpu().numpy()):
                    collision = True
                    break
            if collision:
                break
        if collision:
            lengths = 0
            min_distance = 0    
        collision = 1.0 if collision else 0.0
        smoothness, curvature_score = path_smoothness(Y_pred.cpu().numpy()[0][:30])
        
        target = X_batch['target'][0, :2]
        distances = torch.norm(Y_pred[0] - target, dim=1)
        min_distance = distances.min().item()
        # if collision == 0.0 and dist_violation > 0.01:
        #     print("Warning: No collision detected but distance violation exists. This may indicate a potential issue in the score computation.")
        
        return {
            'length': lengths,
            'collision': collision,
            'smoothness': smoothness,
            'curvature': curvature_score,
            'min_distance': min_distance,
            'dist_violation': dist_violation.item()
        }

    @staticmethod
    def _copy_batch_to_cpu(X_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Keep a detached copy without retaining GPU memory or an autograd graph."""
        return {key: value.detach().cpu().clone() for key, value in X_batch.items()}

    @staticmethod
    def _copy_batch_to_device(X_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {key: value.to(DEVICE, non_blocking=True) for key, value in X_batch.items()}

    @staticmethod
    def _test_sample_id(data_loader: DataLoader, loader_index: int) -> int:
        """Return the source-dataset index when the test set is a Subset."""
        dataset = data_loader.dataset
        if isinstance(dataset, Subset):
            return int(dataset.indices[loader_index])
        return int(loader_index)

    def _diagnostic_stage_metrics(
        self,
        X_batch: Dict[str, torch.Tensor],
        raw_output: torch.Tensor,
        projected_output: torch.Tensor,
        actual_depth: int,
        timings: Dict[str, Optional[float]],
    ) -> Dict[str, object]:
        """Compute the same path metrics used by the normal test for one stage."""
        score = self.compute_score(X_batch, projected_output)
        raw_score = self.compute_score(X_batch, raw_output)
        residuals = self.constraint_func_stage2(X_batch, projected_output)
        residual_max = float(residuals.max().item())
        trajectory_steps = projected_output.shape[1] // self.config['N_dim']
        collision_residual_max = float(
            residuals[:, :trajectory_steps].max().item()
        )
        raw_xy = raw_output.view(raw_output.size(0), -1, self.config['N_dim'])[:, :, :2]
        projected_xy = projected_output.view(
            projected_output.size(0), -1, self.config['N_dim']
        )[:, :, :2]
        proj_distance = float(
            torch.mean(torch.norm(projected_xy - raw_xy, dim=2)).item()
        )
        metrics = {
            'average_time': timings.get('average_time'),
            'network_time': timings.get('network_time'),
            'initialization_time': timings.get('initialization_time'),
            'projection_time': timings.get('projection_time'),
            'jacobian_time': timings.get('jacobian_time'),
            'cholesky_time': timings.get('cholesky_time'),
            'linear_solve_time': timings.get('linear_solve_time'),
            'collision_rate': score['collision'],
            'average_length': score['length'],
            'smoothness': score['smoothness'],
            'curvature': score['curvature'],
            'min_distance': score['min_distance'],
            'dist_violation': score['dist_violation'],
            'actual_depth': int(actual_depth),
            'proj_distance': proj_distance,
            'constraint_residual_max': residual_max,
            # This intentionally matches AdaNPTest's existing non-absolute stop rule.
            'reached_stopping_tolerance': residual_max < self.config['inference_tol'],
            # Use the existing slack-form collision residual directly.
            'differentiable_collision_residual_max': collision_residual_max,
            'differentiable_collision_constraint_satisfied': (
                collision_residual_max <= 0.0
            ),
            'exact_collision': bool(score['collision']),
        }
        return {
            'metrics': metrics,
            'raw_metrics': {
                'collision_rate': raw_score['collision'],
                'average_length': raw_score['length'],
                'smoothness': raw_score['smoothness'],
                'curvature': raw_score['curvature'],
                'min_distance': raw_score['min_distance'],
                'dist_violation': raw_score['dist_violation'],
            },
        }

    def _run_diagnostic_projection(
        self,
        projection: AdaNPTest,
        X_batch: Dict[str, torch.Tensor],
        raw_output: torch.Tensor,
        initial_output: torch.Tensor,
        network_time: float,
        initialization_time: float,
        profile_timing: bool,
    ) -> Tuple[torch.Tensor, int, Dict[str, Optional[float]]]:
        _synchronize_device()
        projection_start = time.perf_counter()
        projected_output, actual_depth = projection(
            X_batch, initial_output, self.constraint_func_stage2
        )
        _synchronize_device()
        projection_time = time.perf_counter() - projection_start
        timings = {
            'network_time': network_time,
            'initialization_time': initialization_time,
            'projection_time': projection_time,
            'average_time': network_time + initialization_time + projection_time,
            'jacobian_time': None,
            'cholesky_time': None,
            'linear_solve_time': None,
        }
        if profile_timing:
            _, _, profile = projection(
                X_batch, initial_output, self.constraint_func_stage2, profile=True
            )
            timings.update(profile)
        return projected_output, actual_depth, timings

    def _diagnostic_raw_stage_metrics(
        self,
        X_batch: Dict[str, torch.Tensor],
        raw_output: torch.Tensor,
        network_time: float,
    ) -> Dict[str, object]:
        """Compute path metrics for an overfit output without projection."""
        score = self.compute_score(X_batch, raw_output)
        metrics = {
            'average_time': network_time,
            'network_time': network_time,
            'initialization_time': 0.0,
            'projection_time': 0.0,
            'jacobian_time': None,
            'cholesky_time': None,
            'linear_solve_time': None,
            'collision_rate': score['collision'],
            'average_length': score['length'],
            'smoothness': score['smoothness'],
            'curvature': score['curvature'],
            'min_distance': score['min_distance'],
            'dist_violation': score['dist_violation'],
            'actual_depth': 0,
            'proj_distance': 0.0,
            'constraint_residual_max': None,
            'reached_stopping_tolerance': None,
            'differentiable_collision_residual_max': None,
            'differentiable_collision_constraint_satisfied': None,
            'exact_collision': bool(score['collision']),
        }
        return {'metrics': metrics, 'raw_metrics': dict(metrics)}

    def _overfit_one_sample(
        self, X_batch: Dict[str, torch.Tensor]
    ) -> Tuple[nn.Module, Dict[str, object]]:
        """Train a freshly initialized network for one failed sample."""
        overfit_model = create_model(self.config, device=DEVICE)
        overfit_model.train()
        steps = int(self.config.get('diag_overfit_steps', 500))
        learning_rate = float(self.config.get('diag_overfit_lr', self.config['lr']))
        endpoint_weight = float(
            self.config.get('diag_overfit_endpoint_weight', 5.0)
        )
        optimizer = optim.Adam(
            overfit_model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.get('weight_decay', 0.0),
        )
        initial_loss = None
        final_loss = None
        initial_map_loss = None
        final_map_loss = None
        initial_soft_constraint_loss = None
        final_soft_constraint_loss = None
        initial_endpoint_loss = None
        final_endpoint_loss = None
        completed_steps = 0
        numerical_failure = None
        start = time.perf_counter()
        with torch.enable_grad():
            for step in range(steps):
                optimizer.zero_grad()
                raw_output = overfit_model(X_batch)
                loss_map = obj_fn(X_batch, raw_output, config=self.config)
                raw_output_view = raw_output.view(
                    raw_output.size(0), -1, self.config['N_dim']
                )
                xy_heading = xy2xy_heading(raw_output_view[:, :, :2])
                loss_soft = soft_constraints(
                    xy_heading,
                    self.config['obs_constraints_weight'],
                    X_batch['obstacles_vertices'],
                )
                endpoint = raw_output_view[:, -1, :2]
                target = X_batch['target'][:, :2]
                loss_endpoint = self.loss_func(endpoint, target)
                loss = loss_map + loss_soft + endpoint_weight * loss_endpoint
                if initial_loss is None:
                    initial_loss = float(loss.detach().item())
                    initial_map_loss = float(loss_map.detach().item())
                    initial_soft_constraint_loss = float(loss_soft.detach().item())
                    initial_endpoint_loss = float(loss_endpoint.detach().item())
                if not torch.isfinite(loss):
                    numerical_failure = f'nonfinite loss at step {step}'
                    break
                loss.backward()
                gradients_finite = all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all()
                    for parameter in overfit_model.parameters()
                )
                if not gradients_finite:
                    numerical_failure = f'nonfinite gradients at step {step}'
                    break
                optimizer.step()
                completed_steps = step + 1
                final_loss = float(loss.detach().item())
                final_map_loss = float(loss_map.detach().item())
                final_soft_constraint_loss = float(loss_soft.detach().item())
                final_endpoint_loss = float(loss_endpoint.detach().item())
        _synchronize_device()
        training_time = time.perf_counter() - start
        overfit_model.eval()
        return overfit_model, {
            'requested_steps': steps,
            'completed_steps': completed_steps,
            'learning_rate': learning_rate,
            'endpoint_weight': endpoint_weight,
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'initial_map_loss': initial_map_loss,
            'final_map_loss': final_map_loss,
            'initial_soft_constraint_loss': initial_soft_constraint_loss,
            'final_soft_constraint_loss': final_soft_constraint_loss,
            'initial_endpoint_loss': initial_endpoint_loss,
            'final_endpoint_loss': final_endpoint_loss,
            'training_time': training_time,
            'numerical_failure': numerical_failure,
            'model_initialization': 'fresh_random_initialization',
            'objective': (
                'global_supervision_plus_soft_constraints_plus_endpoint'
            ),
        }

    def _visualize_diagnostic_pair(
        self,
        X_batch: Dict[str, torch.Tensor],
        before_output: torch.Tensor,
        after_output: torch.Tensor,
        output_dir: str,
        sample_id: int,
        before_label: str,
        after_label: str,
    ) -> Dict[str, object]:
        """Render and name a before/after pair with the existing paper visualizer."""
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = {}
        errors = {}
        for label, output in (
            (before_label, before_output),
            (after_label, after_output),
        ):
            temporary_dir = os.path.join(
                output_dir, f'.sample_{sample_id}_{label}_tmp'
            )
            shutil.rmtree(temporary_dir, ignore_errors=True)
            os.makedirs(temporary_dir, exist_ok=True)
            try:
                trajectories = output.view(
                    output.size(0), -1, self.config['N_dim']
                )[:, :, :2]
                # The visualizer reports its internal temporary filename. Hide
                # that message and report the final renamed file below.
                with contextlib.redirect_stdout(io.StringIO()):
                    visualize_data_batch_paper2(
                        X_batch, trajectories, save_path=temporary_dir
                    )
                generated_files = sorted(
                    filename for filename in os.listdir(temporary_dir)
                    if filename.lower().endswith('.pdf')
                )
                if not generated_files:
                    raise RuntimeError(
                        'visualize_data_batch_paper2 did not produce a PDF'
                    )
                target_path = os.path.join(
                    output_dir, f'sample_{sample_id}_{label}.pdf'
                )
                os.replace(
                    os.path.join(temporary_dir, generated_files[0]), target_path
                )
                saved_paths[label] = target_path
                print(f'Saved diagnostic visualization: {target_path}')
            except Exception as error:
                errors[label] = f'{type(error).__name__}: {error}'
                print(
                    f'Failed to save diagnostic visualization for sample '
                    f'{sample_id} ({label}): {errors[label]}'
                )
            finally:
                shutil.rmtree(temporary_dir, ignore_errors=True)
        return {'files': saved_paths, 'errors': errors}

    def _run_failure_diagnosis(
        self,
        candidates: list,
        total_samples: int,
        profile_timing: bool,
    ) -> str:
        diag_i_max = int(
            self.config.get('DIAG_I_MAX', self.config.get('diag_i_max', 200))
        )
        enable_overfit = bool(
            self.config.get(
                'ENABLE_SAMPLE_OVERFIT',
                self.config.get('enable_sample_overfit', True),
            )
        )
        diag_projection = AdaNPTest(
            max_depth=diag_i_max,
            tol=self.config['inference_tol'],
            w_traj=self.config.get('w_traj', 5.0),
            w_slack=self.config.get('w_slack', 1.0),
            damping=self.config.get('damping', 1e-4),
        )
        output_dir = self.log_dir or self.save_dir or '.'
        iter_img_dir = os.path.join(output_dir, 'iter_img')
        overfit_img_dir = os.path.join(output_dir, 'over_fit_img')
        os.makedirs(iter_img_dir, exist_ok=True)
        os.makedirs(overfit_img_dir, exist_ok=True)
        counts = {
            'surrogate_mismatch': 0,
            'iteration_limited': 0,
            'prediction_route_limited': 0,
            'unresolved': 0,
        }
        records = []
        diagnosis_bar = tqdm.tqdm(candidates, desc='Failure diagnosis')
        for candidate in diagnosis_bar:
            X_batch = self._copy_batch_to_device(candidate['input'])
            raw_default = candidate['default_raw_output'].to(DEVICE)
            initial_default = candidate['default_initial_output'].to(DEVICE)
            projected_default = candidate['imax_50_projected_output'].to(DEVICE)
            default_stage = self._diagnostic_stage_metrics(
                X_batch, raw_default, projected_default,
                candidate['imax_50_actual_iterations'],
                candidate['default_timings'],
            )
            candidate['imax_50'] = default_stage
            candidate['imax_200_projected_output'] = None
            candidate['imax_200_actual_iterations'] = None
            candidate['imax_200'] = {'executed': False, 'metrics': None, 'raw_metrics': None}
            candidate['overfit_raw_output'] = None
            candidate['overfit_projected_output'] = None
            candidate['overfit_actual_iterations'] = None
            candidate['sample_overfit'] = {
                'executed': False, 'training': None,
                'metrics': None, 'raw_metrics': None,
            }

            if (
                default_stage['metrics']['reached_stopping_tolerance']
                and default_stage['metrics'][
                    'differentiable_collision_constraint_satisfied'
                ]
            ):
                failure_type = 'surrogate_mismatch'
            else:
                with torch.no_grad():
                    projected_200, depth_200, timings_200 = self._run_diagnostic_projection(
                        diag_projection, X_batch, raw_default, initial_default,
                        candidate['default_timings']['network_time'],
                        candidate['default_timings']['initialization_time'],
                        profile_timing,
                    )
                    stage_200 = self._diagnostic_stage_metrics(
                        X_batch, raw_default, projected_200, depth_200, timings_200
                    )
                candidate['imax_200_projected_output'] = projected_200.detach().cpu()
                candidate['imax_200_actual_iterations'] = int(depth_200)
                candidate['imax_200'] = {'executed': True, **stage_200}
                if not stage_200['metrics']['exact_collision']:
                    failure_type = 'iteration_limited'
                    candidate['visualizations'] = self._visualize_diagnostic_pair(
                        X_batch,
                        projected_default,
                        projected_200,
                        iter_img_dir,
                        candidate['sample_id'],
                        'Imax50',
                        'Imax200',
                    )
                elif enable_overfit:
                    overfit_model, training_info = self._overfit_one_sample(X_batch)
                    with torch.no_grad():
                        _synchronize_device()
                        network_start = time.perf_counter()
                        overfit_raw = overfit_model(X_batch)
                        _synchronize_device()
                        overfit_network_time = time.perf_counter() - network_start
                        overfit_stage = self._diagnostic_raw_stage_metrics(
                            X_batch, overfit_raw, overfit_network_time
                        )
                    candidate['overfit_raw_output'] = overfit_raw.detach().cpu()
                    candidate['overfit_projected_output'] = None
                    candidate['overfit_actual_iterations'] = 0
                    candidate['sample_overfit'] = {
                        'executed': True,
                        'projection_executed': False,
                        'training': training_info,
                        **overfit_stage,
                    }
                    failure_type = (
                        'prediction_route_limited'
                        if not overfit_stage['metrics']['exact_collision']
                        else 'unresolved'
                    )
                    if failure_type == 'prediction_route_limited':
                        candidate['visualizations'] = (
                            self._visualize_diagnostic_pair(
                                X_batch,
                                projected_200,
                                overfit_raw,
                                overfit_img_dir,
                                candidate['sample_id'],
                                'before_overfit',
                                'after_overfit',
                            )
                        )
                    del overfit_model
                else:
                    candidate['sample_overfit']['skip_reason'] = (
                        'enable_sample_overfit=False'
                    )
                    failure_type = 'unresolved'

            candidate['failure_type'] = failure_type
            counts[failure_type] += 1
            # All saved tensors are detached CPU tensors.
            candidate.pop('default_timings', None)
            records.append(candidate)

        failure_count = len(candidates)
        summary = {
            'total_test_samples': total_samples,
            'default_exact_collision_failures': failure_count,
            'default_exact_collision_failure_rate': (
                failure_count / total_samples if total_samples else 0.0
            ),
        }
        for failure_type, count in counts.items():
            summary[f'{failure_type}_count'] = count
            summary[f'{failure_type}_proportion'] = (
                count / failure_count if failure_count else 0.0
            )

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'failure_diagnosis.pt')
        payload = {
            'summary': summary,
            'samples': records,
            'settings': {
                'default_i_max': self.config['max_depth'],
                'diag_i_max': diag_i_max,
                'inference_tol': self.config['inference_tol'],
                'enable_sample_overfit': enable_overfit,
                'sample_overfit_projection_executed': False,
                'slack_initialization': self.config.get('slack_initialization', 'learned'),
                'w_traj': self.config.get('w_traj', 5.0),
                'w_slack': self.config.get('w_slack', 1.0),
                'damping': self.config.get('damping', 1e-4),
            },
        }
        torch.save(payload, output_path)
        print('=== Failure Diagnosis Summary ===')
        print(f"Total test samples: {total_samples}")
        print(f"Default exact-collision failures: {failure_count}")
        for failure_type, count in counts.items():
            proportion = summary[f'{failure_type}_proportion']
            print(f'{failure_type}: {count} ({proportion:.2%} of failures)')
        print(f'Failure diagnosis saved to {output_path}')
        return output_path

    def _save_failure_sample_ids(self, candidates: list, total_samples: int) -> str:
        """Persist the default-test collision set before further diagnosis."""
        output_dir = self.log_dir or self.save_dir or '.'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'failure_sample_ids.pt')
        payload = {
            'sample_ids': [candidate['sample_id'] for candidate in candidates],
            'test_loader_indices': [
                candidate['test_loader_index'] for candidate in candidates
            ],
            'total_test_samples': total_samples,
            'default_exact_collision_failures': len(candidates),
            'default_i_max': self.config['max_depth'],
            'inference_tol': self.config['inference_tol'],
        }
        torch.save(payload, output_path)
        print(f'Failure sample IDs saved to {output_path}')
        return output_path
        

    def test(self, data_loader: DataLoader = None, test_hard: bool = True, result_name: str = None) -> Dict[str, float]:
        if data_loader is None:
            data_loader = self.test_loader

        test_metrics = {
            'average_time': 0.0, 'network_time': 0.0,
            'initialization_time': 0.0, 'projection_time': 0.0,
            'jacobian_time': 0.0, 'cholesky_time': 0.0,
            'linear_solve_time': 0.0, 'collision_rate': 0.0,
            'average_length': 0.0, 'smoothness': 0.0, 'curvature': 0.0,
            'min_distance': 0.0, 'dist_violation': 0.0,
            'actual_depth': 0.0, 'proj_distance': 0.0,
        }
        profile_timing = self.config.get('profile_timing', False)
        timing_keys = (
            'average_time', 'network_time', 'initialization_time',
            'projection_time',
        )
        if profile_timing:
            timing_keys += (
                'jacobian_time', 'cholesky_time', 'linear_solve_time',
            )
        timing_samples = {key: [] for key in timing_keys}
        self.model.eval()
        total_samples = 0
        enable_failure_diagnosis = test_hard and bool(
            self.config.get(
                'ENABLE_FAILURE_DIAGNOSIS',
                self.config.get('enable_failure_diagnosis', False),
            )
        )
        diagnosis_candidates = [] if enable_failure_diagnosis else None
        # warm up
        with torch.no_grad():
            warm_num = 10
            for X_batch in data_loader:
                for key in X_batch:
                    X_batch[key] = X_batch[key].to(DEVICE, non_blocking=True)
                Y_pred = self.model(X_batch)
                Y_initial = self.adanp.initialize_output(
                    X_batch, Y_pred, self.constraint_func_stage2
                )
                Y_proj, actual_depth = self.adanp_test(
                    X_batch, Y_initial, self.constraint_func_stage2
                )
                warm_num -= 1
                if warm_num <=0:
                    break
        nocollision_samples = 0
        with torch.no_grad():
            test_bar = tqdm.tqdm(data_loader, desc="Testing")
            for loader_index, X_batch in enumerate(test_bar):
                for key in X_batch:
                    X_batch[key] = X_batch[key].to(DEVICE, non_blocking=True)
                _synchronize_device()
                start_time = time.perf_counter()

                network_start = time.perf_counter()
                Y_pred = self.model(X_batch)
                _synchronize_device()
                network_time = time.perf_counter() - network_start

                initialization_start = time.perf_counter()
                Y_initial = self.adanp.initialize_output(
                    X_batch, Y_pred, self.constraint_func_stage2
                )
                _synchronize_device()
                initialization_time = time.perf_counter() - initialization_start

                projection_start = time.perf_counter()
                Y_proj, actual_depth = self.adanp_test(
                    X_batch, Y_initial, self.constraint_func_stage2
                )
                _synchronize_device()
                projection_time = time.perf_counter() - projection_start
                end_time = time.perf_counter()

                profile = None
                if profile_timing:
                    # Run the same projection once more outside the end-to-end
                    # timer to isolate its GPU kernels without contaminating
                    # deployment latency.
                    _, _, profile = self.adanp_test(
                        X_batch, Y_initial, self.constraint_func_stage2,
                        profile=True,
                    )
                Y_pred = Y_pred.view(Y_pred.size(0), -1, self.config['N_dim'])
                Y_pred_xy = Y_pred[:, :, :2]
                Y_proj = Y_proj.view(Y_proj.size(0), -1, self.config['N_dim'])
                Y_proj_xy = Y_proj[:, :, :2]
                proj_distance = torch.mean(torch.norm(Y_proj_xy - Y_pred_xy, dim=2)).item()
                # torch.cuda.synchronize()
                sample_timings = {
                    'average_time': end_time - start_time,
                    'network_time': network_time,
                    'initialization_time': initialization_time,
                    'projection_time': projection_time,
                }
                if profile_timing:
                    sample_timings.update({
                        'jacobian_time': profile['jacobian_time'],
                        'cholesky_time': profile['cholesky_time'],
                        'linear_solve_time': profile['linear_solve_time'],
                    })
                for key, value in sample_timings.items():
                    test_metrics[key] += value
                    timing_samples[key].append(value)
                score_metrics = self.compute_score(X_batch, Y_proj)
                test_metrics['average_length'] += score_metrics['length']
                test_metrics['collision_rate'] += score_metrics['collision']
                test_metrics['smoothness'] += score_metrics['smoothness']
                test_metrics['curvature'] += score_metrics['curvature']
                test_metrics['min_distance'] += score_metrics['min_distance']
                test_metrics['dist_violation'] += score_metrics['dist_violation']
                test_metrics['actual_depth'] += actual_depth
                test_metrics['proj_distance'] += proj_distance
                total_samples += 1
                nocollision_samples += (1.0 - score_metrics['collision'])
                if enable_failure_diagnosis and score_metrics['collision']:
                    diagnosis_candidates.append({
                        'sample_id': self._test_sample_id(data_loader, loader_index),
                        'test_loader_index': loader_index,
                        'input': self._copy_batch_to_cpu(X_batch),
                        'default_raw_output': Y_pred.reshape(Y_pred.size(0), -1).detach().cpu(),
                        'default_initial_output': Y_initial.detach().cpu(),
                        'imax_50_projected_output': Y_proj.reshape(Y_proj.size(0), -1).detach().cpu(),
                        'imax_50_actual_iterations': int(actual_depth),
                        'default_timings': {
                            **sample_timings,
                            'jacobian_time': (
                                profile['jacobian_time'] if profile_timing else None
                            ),
                            'cholesky_time': (
                                profile['cholesky_time'] if profile_timing else None
                            ),
                            'linear_solve_time': (
                                profile['linear_solve_time'] if profile_timing else None
                            ),
                        },
                    })

        if nocollision_samples > 0:
            test_metrics['average_length'] /= nocollision_samples
            test_metrics['min_distance'] /= nocollision_samples
        else:
            # A cached failure-only subset can legitimately contain no
            # collision-free samples, so these conditional means are undefined.
            test_metrics['average_length'] = float('nan')
            test_metrics['min_distance'] = float('nan')
        
        for key in timing_keys:
            test_metrics[key] /= total_samples
            values = timing_samples[key]
            test_metrics[f'{key}_std'] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
        test_metrics['collision_rate'] /= total_samples
        test_metrics['smoothness'] /= total_samples
        test_metrics['dist_violation'] /= total_samples
        test_metrics['curvature'] /= total_samples
        test_metrics['actual_depth'] /= total_samples
        test_metrics['proj_distance'] /= total_samples
        
        print("=== Test Results ===")
        print(f"End-to-end Time: {test_metrics['average_time'] * 1000:.4f} ± {test_metrics['average_time_std'] * 1000:.4f} ms")
        print(f"Network Time: {test_metrics['network_time'] * 1000:.4f} ± {test_metrics['network_time_std'] * 1000:.4f} ms")
        print(f"Initialization Time: {test_metrics['initialization_time'] * 1000:.4f} ± {test_metrics['initialization_time_std'] * 1000:.4f} ms")
        print(f"Projection Time: {test_metrics['projection_time'] * 1000:.4f} ± {test_metrics['projection_time_std'] * 1000:.4f} ms")
        if profile_timing:
            print(f"  Jacobian Construction: {test_metrics['jacobian_time'] * 1000:.4f} ± {test_metrics['jacobian_time_std'] * 1000:.4f} ms")
            print(f"  Cholesky Decomposition: {test_metrics['cholesky_time'] * 1000:.4f} ± {test_metrics['cholesky_time_std'] * 1000:.4f} ms")
            print(f"  Linear Solve and Update: {test_metrics['linear_solve_time'] * 1000:.4f} ± {test_metrics['linear_solve_time_std'] * 1000:.4f} ms")
        print(f"Test Average Length: {test_metrics['average_length']:.4f}")
        print(f"Test Collision Rate: {test_metrics['collision_rate']:.4f}")
        print(f"Test Smoothness: {test_metrics['smoothness']:.4f}")
        print(f"Test Curvature: {test_metrics['curvature']:.4f}")
        print(f"Test Minimum Distance to target: {test_metrics['min_distance']:.4f}")
        print(f"Test Distance Violation: {test_metrics['dist_violation']:.4f}")
        print(f"Test Actual Depth: {test_metrics['actual_depth']:.4f}")
        print(f"Test Projection Distance: {test_metrics['proj_distance']:.4f}")

        file_name = result_name or (
            'test_results_hard.txt' if test_hard else 'test_results_soft.txt'
        )
        # Save test results to a file
        results_file = os.path.join(self.log_dir, file_name) if self.log_dir is not None else file_name
        results_dir = os.path.dirname(results_file)
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
        with open(results_file, 'w') as f:
            f.write("=== Test Results ===\n")
            f.write(f"End-to-end Time: {test_metrics['average_time'] * 1000:.4f} ± {test_metrics['average_time_std'] * 1000:.4f} ms\n")
            f.write(f"Network Time: {test_metrics['network_time'] * 1000:.4f} ± {test_metrics['network_time_std'] * 1000:.4f} ms\n")
            f.write(f"Initialization Time: {test_metrics['initialization_time'] * 1000:.4f} ± {test_metrics['initialization_time_std'] * 1000:.4f} ms\n")
            f.write(f"Projection Time: {test_metrics['projection_time'] * 1000:.4f} ± {test_metrics['projection_time_std'] * 1000:.4f} ms\n")
            if profile_timing:
                f.write(f"  Jacobian Construction: {test_metrics['jacobian_time'] * 1000:.4f} ± {test_metrics['jacobian_time_std'] * 1000:.4f} ms\n")
                f.write(f"  Cholesky Decomposition: {test_metrics['cholesky_time'] * 1000:.4f} ± {test_metrics['cholesky_time_std'] * 1000:.4f} ms\n")
                f.write(f"  Linear Solve and Update: {test_metrics['linear_solve_time'] * 1000:.4f} ± {test_metrics['linear_solve_time_std'] * 1000:.4f} ms\n")
            f.write(f"Test Average Length: {test_metrics['average_length']:.4f}\n")
            f.write(f"Test Collision Rate: {test_metrics['collision_rate']:.4f}\n")
            f.write(f"Test Smoothness: {test_metrics['smoothness']:.4f}\n")
            f.write(f"Test Curvature: {test_metrics['curvature']:.4f}\n")
            f.write(f"Test Minimum Distance to target: {test_metrics['min_distance']:.4f}\n")
            f.write(f"Test Distance Violation: {test_metrics['dist_violation']:.4f}\n")
            f.write(f"Test Actual Depth: {test_metrics['actual_depth']:.4f}\n")
            f.write(f"Test Projection Distance: {test_metrics['proj_distance']:.4f}\n")
        print(f'Test results saved to {results_file}')
        if enable_failure_diagnosis:
            diagnosis_total_samples = int(
                self.config.get(
                    'failure_diagnosis_total_test_samples', total_samples
                )
            )
            self._save_failure_sample_ids(
                diagnosis_candidates, diagnosis_total_samples
            )
            self._run_failure_diagnosis(
                diagnosis_candidates, diagnosis_total_samples, profile_timing
            )
        return

    def evaluate_stage1(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluates the model on a validation or test set."""
        eval_metrics = {'total_loss': 0.0, 'map_loss': 0.0, 'constraint_residuals': 0.0,'loss_cons_func': 0.0}
        self.model.eval()
        with torch.no_grad():
            for X_batch in data_loader:
                for key in X_batch:
                    X_batch[key] = X_batch[key].to(DEVICE, non_blocking=True)
                Y_pred = self.model(X_batch)
                Y_pred = Y_pred.view(Y_pred.size(0), -1, self.config['N_dim'])  # (B, N, 2)
                loss_map = obj_fn(X_batch, Y_pred, config=self.config)
                xy_pred = Y_pred[:,:,:2]
                xy_heading = xy2xy_heading(xy_pred)  # (B, N, 3)
                
                loss_soft = soft_constraints(xy_heading, self.config['obs_constraints_weight'], X_batch['obstacles_vertices'])
                loss_cons_func = self.constraint_func_stage2(X_batch, Y_pred).mean()
                
                eval_metrics['total_loss'] += loss_map.item() + loss_soft.item() + loss_cons_func.item()
                eval_metrics['map_loss'] += loss_map.item()
                eval_metrics['constraint_residuals'] += loss_soft.item()
                eval_metrics['loss_cons_func'] += loss_cons_func.item()

        num_batches = len(data_loader)
        for key in eval_metrics:
            eval_metrics[key] /= num_batches
            
        return eval_metrics
    
    def evaluate_stage2(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluates the model on a validation or test set."""
        eval_metrics = {'map_loss': 0.0, 'loss_soft_pred': 0.0,'loss_soft_proj': 0.0, 'loss_cons_func': 0.0, 'loss_func_proj': 0.0, 'actual_depth': 0.0}
        self.model.eval()
        with torch.no_grad():
            for X_batch in data_loader:
                for key in X_batch:
                    X_batch[key] = X_batch[key].to(DEVICE, non_blocking=True)
                
                Y_pred = self.model(X_batch) #(B, 280)
                loss_map = obj_fn(X_batch, Y_pred, config=self.config)
                Y_proj, depth, actual_depth  = self.adanp(X_batch, Y_pred, self.constraint_func_stage2) #(B, 280)
                actual_depth = actual_depth.detach().cpu().numpy()
                
                B =Y_pred.size(0)
                
                Y_pred_ = Y_pred.view(B, -1, 7)  # (B, N, 7)
                Y_proj_ = Y_proj.view(B, -1, 7)  # (B, N, 7)
                
                xy_pred = Y_pred_[:,:,:2]
                xy_proj = Y_proj_[:,:,:2]
                
                xy_heading_pred = xy2xy_heading(xy_pred)  # (B, N, 3)
                xy_heading_proj = xy2xy_heading(xy_proj)  # (B, N, 3)
                
                
                loss_soft_pred = soft_constraints(xy_heading_pred, self.config['obs_constraints_weight'], X_batch['obstacles_vertices'])
                loss_soft_proj = soft_constraints(xy_heading_proj, self.config['obs_constraints_weight'], X_batch['obstacles_vertices'])
                
                loss_func_pred = self.constraint_func_stage2(X_batch, Y_pred).mean()
                loss_func_proj = self.constraint_func_stage2(X_batch, Y_proj).mean()
                
                eval_metrics['map_loss'] += loss_map.item()
                eval_metrics['loss_soft_pred'] += loss_soft_pred.item()
                eval_metrics['loss_soft_proj'] += loss_soft_proj.item()
                eval_metrics['loss_cons_func'] += loss_func_pred.item()
                eval_metrics['loss_func_proj'] += loss_func_proj.item()
                eval_metrics['actual_depth'] += actual_depth.mean()

        num_batches = len(data_loader)
        for key in eval_metrics:
            eval_metrics[key] /= num_batches
            
        return eval_metrics
    
    def test_visualization(self, save_path: str = None):
        """Generates visualizations for the test set."""
        if self.test_loader is None:
            print("No test loader provided for visualization.")
            return
        if save_path is None:
            save_path = self.log_dir
        os.makedirs(save_path, exist_ok=True)
        self.model.eval()
        with torch.no_grad():
            for i, X_batch in enumerate(self.test_loader):
                for key in X_batch:
                    X_batch[key] = X_batch[key].to(DEVICE, non_blocking=True)
                Y_pred = self.model(X_batch)
                Y_proj, depth, actual_depth = self.adanp._original_forward(X_batch, Y_pred, self.constraint_func_stage2)
                Y_pred_ = Y_pred.view(Y_pred.size(0), -1, self.config['N_dim'])  # (B, N, 2)
                Y_proj_ = Y_proj.view(Y_proj.size(0), -1, self.config['N_dim'])  # (B, N, 2)
                xy_pred = Y_pred_[:,:,:2]
                xy_proj = Y_proj_[:,:,:2]
                
                trajectories = xy_pred
                visualize_data_batch_paper2(X_batch, trajectories, save_path=save_path)
                trajectories = xy_proj
                visualize_data_batch_paper2(X_batch, trajectories, save_path=save_path)
                
                # break  # Visualize only the first batch for brevity


    def _save_model(self, epoch: int):
        """Saves the model checkpoint."""
        if self.save_dir is not None:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'config': self.config
            }
            torch.save(checkpoint, f'{self.save_dir}/epoch_{epoch}.pth')
            print(f'Model checkpoint saved at epoch {epoch} to {self.save_dir}')
            
    def save_path_data(self, data_loader: DataLoader = None, path_data_dir=None) -> Dict[str, float]:
        self.model.eval()
        os.makedirs(path_data_dir, exist_ok=True)
        if data_loader is None:
            data_loader = self.test_loader
        # 3. Main test loop
        with torch.no_grad():
            for batch_idx, X_batch in enumerate(data_loader):
                # Move data to the target device
                save_path = os.path.join(path_data_dir, f'batch_{batch_idx}.npy')
                if os.path.exists(save_path):
                    continue
                for key in X_batch:
                    X_batch[key] = X_batch[key].to(DEVICE, non_blocking=True)
                
                # Run model inference
                Y_pred = self.model(X_batch)
                Y_proj, depth, actual_depth = self.adanp._original_forward(X_batch, Y_pred, self.constraint_func_stage2)
                Y_final = Y_proj.view(Y_proj.size(0), -1, 7)  # (B, N, 7)
                Y_final = Y_final[:, :, :2]  # (B, N, 2)
                Y_final_numpy = Y_final[0].cpu().numpy()
                # Save Y_final_numpy
                np.save(save_path, Y_final_numpy)
                print(f"Saved Y_final_numpy for batch {batch_idx}.")
