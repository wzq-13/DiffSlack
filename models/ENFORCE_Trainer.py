import numpy as np
import pickle
import time
import os 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple, Callable, Optional, Dict
import tqdm
import globalvar
from utils.utils import visualize_data_batch, check_polygon_intersection, get_rect_points_vectorized, visualize_data_batch_paper, path_smoothness,visualize_data_batch_paper2
from utils.prob import xy2xy_heading, soft_constraints, xy2xy_heading, compute_kappa_menger, get_safe_circle_centers, h, LSE_max, world_to_grid, get_map_distance, create_enforce_inequality_constraints
from models.utils import create_model, path_clean
from torch.utils.tensorboard import SummaryWriter
from models.ENFORCE import ENFORCEAadaNP, ENFORCEAadaNPTest

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# DEVICE = torch.device("cpu")

def obj_fn(data, y, config):
    distance_map = data['distance_map']  # (batch_size, H, W)
    xy = y.view(y.shape[0], -1, 2)  # (batch_size, N, 2)
    world_x = xy[:, :, 0]  # (B, N)
    world_y = xy[:, :, 1]  # (B, N)
    i, j = world_to_grid(world_x, world_y)  # (B, N)
    distances = get_map_distance(distance_map, i, j, config)  # (B, N)
    map_loss = distances.mean()
    return map_loss

class ENFORCE_Trainer:
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
        self.adanp = ENFORCEAadaNP(max_depth=self.config['max_depth'], tol=self.config['inference_tol'])
        self.adanp_test = ENFORCEAadaNPTest(max_depth=self.config['max_depth'], tol=self.config['inference_tol'])
        self.training_tol = self.config['training_tol']
        self.constraint_func = create_enforce_inequality_constraints()
        
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
            
            
            loss = loss_map + loss_cons * self.config['slack_weight']
            
            loss.backward()
            self.optimizer.step()
            bar.set_postfix(
                loss=f"{loss.item():.4f}",
                loss_cons=f"{loss_cons.item():.4f}",
            )
            epoch_metrics['total_loss'] += loss.item()
            epoch_metrics['loss_map'] += loss_map.item()
            epoch_metrics['loss_soft'] += loss_cons.item()
            
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
            Y_proj, actual_depth, info = self.adanp(X_batch, Y_pred, self.constraint_func)
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
            
            loss_proj = torch.mean((Y_proj.detach() - Y_pred)**2)
            
            loss = loss_map_proj + loss_end + loss_end_pred*5 + loss_cons + loss_proj * self.config['proj_loss_weight']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            actual_depth = actual_depth.detach().cpu().numpy()
            bar.set_postfix(
                loss=f"{loss.item():.4f}",
                loss_map_proj=f"{loss_map_proj.item():.4f}",
                loss_cons=f"{loss_cons.item():.4f}",
                loss_cons_proj=f"{loss_cons_proj.item():.4f}",
                loss_proj=f"{loss_proj.item():.4f}",
                actual_depth =f"{actual_depth.mean():.4f}"
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
                    self.test_visualization(save_path=self.log_dir)
            self.test(self.test_loader, test_hard=False)
        # return #for ablation study, APF+Soft only
            
        del self.train_loader, self.val_loader
        self.train_loader = DataLoader(self.train_dataset, batch_size=int(self.config['batch_size']//2), shuffle=True, pin_memory=True,num_workers=8,persistent_workers=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=int(self.config['batch_size']//2), shuffle=False, pin_memory=True,num_workers=8,persistent_workers=False)
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
                self.test_visualization(save_path=self.log_dir)
            
            if (epoch + 1) % 10 == 0:
                time.sleep(20)
        
        self.test_visualization(save_path=self.log_dir)
        self.test(self.test_loader, test_hard=True)
    

    def compute_score(self, X_batch: torch.Tensor, Y_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Computes score."""
        B = Y_pred.size(0)
        # compute length
        Y_pred = Y_pred.view(B, -1, self.config['N_dim'])
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
        

    def test(self, data_loader: DataLoader = None, test_hard: bool = True, result_name: str = None) -> Dict[str, float]:
        if data_loader is None:
            data_loader = self.test_loader

        test_metrics = {'average_time': 0.0, 'collision_rate':0.0, 'average_length':0.0, 'smoothness':0.0, 'curvature':0.0, 'min_distance':0.0, 'dist_violation': 0.0, 'actual_depth': 0.0, 'proj_distance': 0.0}
        self.model.eval()
        total_samples = 0
        # warm up
        with torch.no_grad():
            warm_num = 10
            for X_batch in data_loader:
                for key in X_batch:
                    X_batch[key] = X_batch[key].to(DEVICE, non_blocking=True)
                Y_pred = self.model(X_batch)
                Y_proj, actual_depth, info =  self.adanp_test(X_batch, Y_pred, self.constraint_func)
                warm_num -= 1
                if warm_num <=0:
                    break
        nocollision_samples = 0
        with torch.no_grad():
            test_bar = tqdm.tqdm(data_loader, desc="Testing")
            for X_batch in test_bar:
                for key in X_batch:
                    X_batch[key] = X_batch[key].to(DEVICE, non_blocking=True)
                # torch.cuda.synchronize()
                start_time = time.time()
                Y_pred = self.model(X_batch)
                # if test_hard:
                Y_proj, actual_depth, info = self.adanp_test(X_batch, Y_pred, self.constraint_func)
                end_time = time.time()
                Y_pred_xy = Y_pred.view(Y_pred.size(0), -1, 2)  # (B, N, 2)
                Y_proj_xy = Y_proj.view(Y_proj.size(0), -1, 2)  # (B, N, 2)
                proj_distance = torch.norm(Y_proj_xy - Y_pred_xy, dim=2).mean().item()
                # torch.cuda.synchronize()
                test_metrics['average_time'] += (end_time - start_time)
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

        test_metrics['average_length'] /= nocollision_samples
        test_metrics['min_distance'] /= nocollision_samples
        
        test_metrics['average_time'] /= total_samples
        test_metrics['collision_rate'] /= total_samples
        test_metrics['smoothness'] /= total_samples
        test_metrics['dist_violation'] /= total_samples
        test_metrics['curvature'] /= total_samples
        test_metrics['actual_depth'] /= total_samples
        test_metrics['proj_distance'] /= total_samples
        
        print("=== Test Results ===")
        print(f"Test Average Time per Batch: {test_metrics['average_time']:.4f} seconds")
        print(f"Test Average Length: {test_metrics['average_length']:.4f}")
        print(f"Test Collision Rate: {test_metrics['collision_rate']:.4f}")
        print(f"Test Smoothness: {test_metrics['smoothness']:.4f}")
        print(f"Test Curvature: {test_metrics['curvature']:.4f}")
        print(f"Test Minimum Distance to target: {test_metrics['min_distance']:.4f}")
        print(f"Test Distance Violation: {test_metrics['dist_violation']:.4f}")
        print(f"Test Actual Depth: {test_metrics['actual_depth']:.4f}")
        print(f"Test Projection Distance: {test_metrics['proj_distance']:.4f}")

        # file_name = result_name or f'ab_test/test_I_noEarlyEnd_{self.config["max_depth"]}.txt'
        file_name = 'test_results_hard.txt' if test_hard else 'test_results_soft.txt'
        # 保存测试结果到文件
        results_file = os.path.join(self.log_dir, file_name) if self.log_dir is not None else file_name
        results_dir = os.path.dirname(results_file)
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
        with open(results_file, 'w') as f:
            f.write("=== Test Results ===\n")
            f.write(f"Test Average Time per Batch: {test_metrics['average_time']:.4f} seconds\n")
            f.write(f"Test Average Length: {test_metrics['average_length']:.4f}\n")
            f.write(f"Test Collision Rate: {test_metrics['collision_rate']:.4f}\n")
            f.write(f"Test Smoothness: {test_metrics['smoothness']:.4f}\n")
            f.write(f"Test Curvature: {test_metrics['curvature']:.4f}\n")
            f.write(f"Test Minimum Distance to target: {test_metrics['min_distance']:.4f}\n")
            f.write(f"Test Distance Violation: {test_metrics['dist_violation']:.4f}\n")
            f.write(f"Test Actual Depth: {test_metrics['actual_depth']:.4f}\n")
            f.write(f"Test Projection Distance: {test_metrics['proj_distance']:.4f}\n")
        print(f'Test results saved to {results_file}')
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
                loss_cons_func = self.constraint_func(X_batch, Y_pred).mean()
                
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
                Y_proj, actual_depth, info  = self.adanp(X_batch, Y_pred, self.constraint_func) #(B, 280)
                actual_depth = actual_depth.detach().cpu().numpy()
                
                B =Y_pred.size(0)
                
                Y_pred_ = Y_pred.view(B, -1, self.config['N_dim'])
                Y_proj_ = Y_proj.view(B, -1, self.config['N_dim'])
                
                xy_pred = Y_pred_[:,:,:2]
                xy_proj = Y_proj_[:,:,:2]
                
                xy_heading_pred = xy2xy_heading(xy_pred)  # (B, N, 3)
                xy_heading_proj = xy2xy_heading(xy_proj)  # (B, N, 3)
                
                
                loss_soft_pred = soft_constraints(xy_heading_pred, self.config['obs_constraints_weight'], X_batch['obstacles_vertices'])
                loss_soft_proj = soft_constraints(xy_heading_proj, self.config['obs_constraints_weight'], X_batch['obstacles_vertices'])
                
                loss_func_pred = self.constraint_func(X_batch, Y_pred).mean()
                loss_func_proj = self.constraint_func(X_batch, Y_proj).mean()
                
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
                Y_proj, actual_depth, info = self.adanp._original_forward(X_batch, Y_pred, self.constraint_func)
                Y_pred_ = Y_pred.view(Y_pred.size(0), -1, self.config['N_dim'])  # (B, N, 2)
                Y_proj_ = Y_proj.view(Y_proj.size(0), -1, self.config['N_dim'])  # (B, N, 2)
                xy_pred = Y_pred_[:,:,:2]
                xy_proj = Y_proj_[:,:,:2]
                
                trajectories = xy_pred
                visualize_data_batch_paper2(X_batch, trajectories, save_path=save_path)
                trajectories = xy_proj
                visualize_data_batch_paper2(X_batch, trajectories, save_path=save_path)
                
                break  # Visualize only the first batch for brevity


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
        # 3. 正式测试循环
        with torch.no_grad():
            for batch_idx, X_batch in enumerate(data_loader):
                # 数据搬运
                save_path = os.path.join(path_data_dir, f'batch_{batch_idx}.npy')
                if os.path.exists(save_path):
                    continue
                for key in X_batch:
                    X_batch[key] = X_batch[key].to(DEVICE, non_blocking=True)
                
                # 模型推理
                Y_pred = self.model(X_batch)
                Y_proj,  actual_depth, info = self.adanp._original_forward(X_batch, Y_pred, self.constraint_func)
                Y_final = Y_proj.view(Y_proj.size(0), -1, self.config['N_dim'])
                Y_final = Y_final[:, :, :2]  # (B, N, 2)
                Y_final_numpy = Y_final[0].cpu().numpy()
                # 保存Y_final_numpy
                np.save(save_path, Y_final_numpy)
                print(f"Saved Y_final_numpy for batch {batch_idx}.")
