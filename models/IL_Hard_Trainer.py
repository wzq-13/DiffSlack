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
from utils.utils import check_polygon_intersection, get_rect_points_vectorized, visualize_data_batch_paper, path_smoothness,visualize_data_batch_paper2, path_clean
from utils.prob import _create_objective_function, obj_fn, xy2xy_heading, soft_constraints, xy2xy_heading
from models.utils import create_model
from torch.utils.tensorboard import SummaryWriter
from models.DiffSlack import AdaNP

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# DEVICE = torch.device("cpu")

class IL_Hard_Trainer:
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
        self.adanp = AdaNP(max_depth=self.config['max_depth'], tol=self.config['inference_tol'])
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
        epoch_metrics = {'total_loss': 0.0, 'loss_BC': 0.0, 'loss_soft': 0.0, 'loss_slack': 0.0}
        self.model.train()
        self.adanp.train()
        bar = tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{self.config['num_epochs_stage1']}")
        for X_batch in bar:
            for key in X_batch:
                X_batch[key] = X_batch[key].to(DEVICE, non_blocking=True)
            Y_true = X_batch['path'].view(-1, 80)
            self.optimizer.zero_grad()
            Y_pred = self.model(X_batch)
            Y_pred_ = Y_pred.view(Y_pred.size(0), -1, 7)  # (B, N, 2)
            xy_pred = Y_pred_[:,:,:2]
            
            xy_heading = xy2xy_heading(xy_pred)  # (B, N, 3)
            
            loss_BC = self.loss_func(xy_pred.reshape(-1, 80), Y_true)
            
            loss_cons = soft_constraints(xy_heading, self.config['obs_constraints_weight'], X_batch['obstacles_vertices'])
            
            loss_slack = self.constraint_func_stage1(X_batch, Y_pred).abs().mean()
            
            loss = loss_BC + loss_cons + loss_slack* self.config['slack_weight']
            
            loss.backward()
            self.optimizer.step()
            bar.set_postfix(
                loss=f"{loss.item():.4f}",
                loss_BC=f"{loss_BC.item():.4f}",
                loss_cons=f"{loss_cons.item():.4f}",
                loss_slack=f"{loss_slack.item():.4f}"
            )
            epoch_metrics['total_loss'] += loss.item()
            epoch_metrics['loss_BC'] += loss_BC.item()
            epoch_metrics['loss_soft'] += loss_cons.item()
            epoch_metrics['loss_slack'] += loss_slack.item()
            
        self.scheduler.step()
        
        num_batches = len(train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
            
        return epoch_metrics
    
    def train_epoch_stage2(self, train_loader: DataLoader, epoch: int):
        """Trains the model for one epoch."""
        epoch_metrics = {'total_loss': 0.0, 'loss_BC': 0.0, 'loss_soft': 0.0, 'loss_proj': 0.0}
        self.model.train()
        self.adanp.train()
        bar = tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{self.config['num_epochs_stage2']}")
        for X_batch in bar:
            for key in X_batch:
                X_batch[key] = X_batch[key].to(DEVICE, non_blocking=True)
            Y_true = X_batch['path'].view(-1, 80)
            self.optimizer.zero_grad()
            Y_pred = self.model(X_batch)
            B = Y_pred.size(0)
            Y_proj, depth, actual_depth = self.adanp(X_batch, Y_pred, self.constraint_func_stage2)
            
            Y_pred_ = Y_pred.view(B, -1, 7)
            Y_proj_ = Y_proj.view(B, -1, 7)
            
            xy_heading = xy2xy_heading(Y_pred_[:,:,:2])  # (B, N, 3)
            end_point = Y_pred_[:, -1, :2]
            target_point = X_batch['target'][:, :2]
            loss_end = self.loss_func(end_point, target_point)
            
            loss_BC = self.loss_func(Y_proj_[:,:,:2].reshape(-1, 80), Y_true)
            
            loss_cons = soft_constraints(xy_heading, self.config['obs_constraints_weight'], X_batch['obstacles_vertices'])
            
            loss_proj = torch.mean((Y_proj.detach() - Y_pred)**2)
            
            loss = loss_BC + loss_proj * self.config['proj_loss_weight'] + loss_cons + loss_end
            loss.backward()
            
            self.optimizer.step()
            bar.set_postfix(
                loss=f"{loss.item():.4f}",
                loss_BC=f"{loss_BC.item():.4f}",
                loss_cons=f"{loss_cons.item():.4f}",
                loss_proj=f"{loss_proj.item():.4f}",
                depth=f"{torch.mean(depth.float()).item():.4f}"
            )
            epoch_metrics['total_loss'] += loss.item()
            epoch_metrics['loss_BC'] += loss_BC.item()
            epoch_metrics['loss_soft'] += loss_cons.item()
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
            self.test(test_hard=False)
        # return #for ablation study, APF+Soft only
            
        del self.train_loader, self.val_loader
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config['batch_size']//3, shuffle=True, pin_memory=True,num_workers=8,persistent_workers=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.config['batch_size']//3, shuffle=False, pin_memory=True,num_workers=8,persistent_workers=False)
        num_epochs_stage2 = self.config['num_epochs_stage2']
        self.config['save_step'] = max(1, self.config['save_step'] // 4)
        
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
                time.sleep(20) # 每10个epoch休息60秒，缓解GPU压力
        
        self.test_visualization(save_path=self.log_dir)
        self.test(test_hard=True)
    

    def compute_score(self, X_batch: torch.Tensor, Y_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Computes score."""
        B = Y_pred.size(0)
        # compute length
        Y_pred = Y_pred.view(B, -1, 7)  # (B, N, 7)
        Y_pred = Y_pred[:, :, :2]  # (B, N, 2)
        Y_cleaned = path_clean(Y_pred[0].detach().cpu().numpy(), X_batch['target'][0, :2].detach().cpu().numpy())  # (N_cleaned, 2)
        lengths = np.sum(np.linalg.norm(np.diff(Y_cleaned, axis=0), axis=1))
        
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
        collision = 1.0 if collision else 0.0
        smoothness, curvature_score = path_smoothness(Y_pred.cpu().numpy()[0][:30])
        
        target = X_batch['target'][0, :2]
        distances = torch.norm(Y_pred[0] - target, dim=1)
        min_distance = distances.min().item()
        
        return {
            'length': lengths.item(),
            'collision': collision,
            'smoothness': smoothness,
            'curvature': curvature_score,
            'min_distance': min_distance,
            'dist_violation': dist_violation.item()
        }
        

    def test(self, data_loader: DataLoader = None, test_hard: bool = False) -> Dict[str, float]:
        if data_loader is None:
            data_loader = self.test_loader

        test_metrics = {'average_time': 0.0, 'collision_rate':0.0, 'average_length':0.0, 'smoothness':0.0, 'curvature':0.0, 'min_distance':0.0, 'dist_violation':0.0}
        self.model.eval()
        total_samples = 0
        
        # warm up
        with torch.no_grad():
            warm_num = 10
            for X_batch in data_loader:
                for key in X_batch:
                    X_batch[key] = X_batch[key].to(DEVICE, non_blocking=True)
                Y_pred = self.model(X_batch)
                warm_num -= 1
                if warm_num <=0:
                    break
        
        with torch.no_grad():
            test_bar = tqdm.tqdm(data_loader, desc="Testing")
            for X_batch in test_bar:
                for key in X_batch:
                    X_batch[key] = X_batch[key].to(DEVICE, non_blocking=True)
                # torch.cuda.synchronize()
                start_time = time.time()
                Y_pred = self.model(X_batch)
                Y_proj, depth, actual_depth = self.adanp._original_forward(X_batch, Y_pred, self.constraint_func_stage2)
                depth = depth.detach().cpu().numpy()
                test_bar.set_postfix(depth=f"{depth[0]:.4f}")
                # torch.cuda.synchronize()
                end_time = time.time()
                test_metrics['average_time'] += (end_time - start_time)
                score_metrics = self.compute_score(X_batch, Y_proj)
                test_metrics['average_length'] += score_metrics['length']
                test_metrics['collision_rate'] += score_metrics['collision']
                test_metrics['smoothness'] += score_metrics['smoothness']
                test_metrics['curvature'] += score_metrics['curvature']
                test_metrics['min_distance'] += score_metrics['min_distance']
                test_metrics['dist_violation'] += score_metrics['dist_violation']
                total_samples += 1

        for key in test_metrics:
            test_metrics[key] /= total_samples
        print("=== Test Results ===")
        print(f"Test Average Time per Batch: {test_metrics['average_time']:.4f} seconds")
        print(f"Test Average Length: {test_metrics['average_length']:.4f}")
        print(f"Test Collision Rate: {test_metrics['collision_rate']:.4f}")
        print(f"Test Smoothness: {test_metrics['smoothness']:.4f}")
        print(f"Test Curvature: {test_metrics['curvature']:.4f}")
        print(f"Test Minimum Distance to Obstacles: {test_metrics['min_distance']:.4f}")
        print(f"Test Distance Violation: {test_metrics['dist_violation']:.4f}")
        
        # 保存测试结果到文件
        results_file = os.path.join(self.log_dir, 'test_results_same.txt') if self.log_dir is not None else 'test_results.txt'
        with open(results_file, 'w') as f:
            f.write("=== Test Results ===\n")
            f.write(f"Test Average Time per Batch: {test_metrics['average_time']:.4f} seconds\n")
            f.write(f"Test Average Length: {test_metrics['average_length']:.4f}\n")
            f.write(f"Test Collision Rate: {test_metrics['collision_rate']:.4f}\n")
            f.write(f"Test Smoothness: {test_metrics['smoothness']:.4f}\n")
            f.write(f"Test Curvature: {test_metrics['curvature']:.4f}\n")
            f.write(f"Test Minimum Distance to Obstacles: {test_metrics['min_distance']:.4f}\n")
            f.write(f"Test Distance Violation: {test_metrics['dist_violation']:.4f}\n")
        print(f'Test results saved to {results_file}')
        return

    def evaluate_stage1(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluates the model on a validation or test set."""
        eval_metrics = {'total_loss': 0.0, 'bc_loss': 0.0, 'constraint_residuals': 0.0,'loss_slack': 0.0}
        self.model.eval()
        self.adanp.eval()
        with torch.no_grad():
            for X_batch in data_loader:
                for key in X_batch:
                    X_batch[key] = X_batch[key].to(DEVICE, non_blocking=True)
                Y_true = X_batch['path'].view(-1, 80)
                Y_pred = self.model(X_batch)
                Y_pred = Y_pred.view(Y_pred.size(0), -1, 7)  # (B, N, 2)
                xy_pred = Y_pred[:,:,:2]
                xy_heading = xy2xy_heading(xy_pred)  # (B, N, 3)
                loss_bc = self.loss_func(xy_pred.reshape(-1, 80), Y_true)
                
                loss_soft = soft_constraints(xy_heading, self.config['obs_constraints_weight'], X_batch['obstacles_vertices'])
                loss_cons_func = self.constraint_func_stage2(X_batch, Y_pred).mean()
                
                eval_metrics['total_loss'] += loss_bc.item() + loss_soft.item() + loss_cons_func.item()
                eval_metrics['bc_loss'] += loss_bc.item()
                eval_metrics['constraint_residuals'] += loss_soft.item()
                eval_metrics['loss_slack'] += loss_cons_func.item()

        num_batches = len(data_loader)
        for key in eval_metrics:
            eval_metrics[key] /= num_batches
        self.adanp.train()
        self.model.train()
        return eval_metrics
    
    def evaluate_stage2(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluates the model on a validation or test set."""
        eval_metrics = {'bc_loss': 0.0, 'loss_soft_pred': 0.0,'loss_soft_proj': 0.0, 'loss_func_pred': 0.0, 'loss_func_proj': 0.0}
        self.model.eval()
        self.adanp.eval()
        with torch.no_grad():
            for X_batch in data_loader:
                for key in X_batch:
                    X_batch[key] = X_batch[key].to(DEVICE, non_blocking=True)
                Y_true = X_batch['path'].view(-1, 80)
                
                Y_pred = self.model(X_batch) #(B, 280)
                
                Y_proj, depth = self.adanp(X_batch, Y_pred, self.constraint_func_stage2) #(B, 280)
                
                B =Y_pred.size(0)
                
                Y_pred_ = Y_pred.view(B, -1, 7)  # (B, N, 7)
                Y_proj_ = Y_proj.view(B, -1, 7)  # (B, N, 7)
                
                xy_pred = Y_pred_[:,:,:2]
                xy_proj = Y_proj_[:,:,:2]
                
                xy_heading_pred = xy2xy_heading(xy_pred)  # (B, N, 3)
                xy_heading_proj = xy2xy_heading(xy_proj)  # (B, N, 3)
                
                loss_bc = self.loss_func(xy_pred.reshape(-1, 80), Y_true)
                
                loss_soft_pred = soft_constraints(xy_heading_pred, self.config['obs_constraints_weight'], X_batch['obstacles_vertices'])
                loss_soft_proj = soft_constraints(xy_heading_proj, self.config['obs_constraints_weight'], X_batch['obstacles_vertices'])
                
                loss_func_pred = self.constraint_func_stage2(X_batch, Y_pred).mean()
                loss_func_proj = self.constraint_func_stage2(X_batch, Y_proj).mean()
                
                eval_metrics['bc_loss'] += loss_bc.item()
                eval_metrics['loss_soft_pred'] += loss_soft_pred.item()
                eval_metrics['loss_soft_proj'] += loss_soft_proj.item()
                eval_metrics['loss_func_pred'] += loss_func_pred.item()
                eval_metrics['loss_func_proj'] += loss_func_proj.item()

        num_batches = len(data_loader)
        for key in eval_metrics:
            eval_metrics[key] /= num_batches
        self.adanp.train()
        self.model.train()
        return eval_metrics
    
    def save_path_data(self, data_loader: DataLoader, path_data_dir='/mnt/sim/carla/carla-ue4-0.9.16/PythonAPI/examples/path_data/IL2') -> Dict[str, float]:
        self.model.eval()
        self.adanp.eval()
        if not os.path.exists(path_data_dir):
            os.makedirs(path_data_dir)
        
        # 3. 正式测试循环
        with torch.no_grad():
            for batch_idx, X_batch in enumerate(data_loader):
                # 数据搬运
                save_path = os.path.join(path_data_dir, f'batch_{batch_idx}.npy')
                # if os.path.exists(save_path):
                #     continue
                for key in X_batch:
                    X_batch[key] = X_batch[key].to(DEVICE, non_blocking=True)
                
                # 模型推理
                Y_final = self.model(X_batch)
                Y_final = Y_final.view(Y_final.size(0), -1, 2)  # (B, N, 2)
                Y_final_numpy = Y_final[0].cpu().numpy()
                # 保存Y_final_numpy
                np.save(save_path, Y_final_numpy)
                print(f"Saved Y_final_numpy for batch {batch_idx}.")
        self.adanp.train()
        self.model.train()
        return
    
    def test_visualization(self, save_path: str = None):
        """Generates visualizations for the test set."""
        if self.test_loader is None:
            print("No test loader provided for visualization.")
            return
        if save_path is None:
            save_path = self.log_dir
        os.makedirs(save_path, exist_ok=True)
        self.model.eval()
        self.adanp.eval()
        with torch.no_grad():
            for i, X_batch in enumerate(self.test_loader):
                for key in X_batch:
                    X_batch[key] = X_batch[key].to(DEVICE, non_blocking=True)
                Y_pred = self.model(X_batch)
                Y_proj, depth, actual_depth = self.adanp._original_forward(X_batch, Y_pred, self.constraint_func_stage2)
                print('--- Visualization ---')
                print(f"Visualization batch {i}, depth:")
                print(depth)
                Y_pred_ = Y_pred.view(Y_pred.size(0), -1, 7)  # (B, N, 2)
                Y_proj_ = Y_proj.view(Y_proj.size(0), -1, 7)  # (B, N, 2)
                xy_pred = Y_pred_[:,:,:2]
                xy_proj = Y_proj_[:,:,:2]
                
                # trajectories = xy_pred
                # visualize_data_batch_paper2(X_batch, trajectories, save_path=save_path)
                trajectories = xy_proj
                visualize_data_batch_paper2(X_batch, trajectories, save_path=save_path)
                
                break  # Visualize only the first batch for brevity
        self.adanp.train()
        self.model.train()

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