import numpy as np
import pickle
import time
import os 
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import csv
import pandas as pd
from typing import Tuple, Callable, Optional, Dict
from torch.utils.tensorboard import SummaryWriter
from torch.func import vmap, jacrev
import globalvar
import matplotlib.pyplot as plt
from utils.utils import visualize_data_batch, check_polygon_intersection, get_rect_points_vectorized, visualize_data_batch_2, path_smoothness, visualize_data_batch_paper2
from utils.prob import _create_objective_function, obj_fn, xy2xy_heading
from models.neural_networks import MLP
from models.utils import create_model, path_clean

RESULT_DIR = './test_hard2_logs'
CSV_FILE_PATH = os.path.join(RESULT_DIR, 'batch_details.csv')
SUMMARY_FILE_PATH = os.path.join(RESULT_DIR, 'final_summary.txt')

class NeuralProjection(nn.Module):
    def __init__(self, n_traj=80, n_slack=200, w_traj=1.0, w_slack=0.1):
        super().__init__()
        self.n_traj = n_traj
        self.n_slack = n_slack
        self.w_traj = w_traj
        self.w_slack = w_slack
        
    def compute_batch_jacobian(self, data, y, constraints_fn):
        def single_constraint_fn(y_single, data_single):
            y_fake_batch = y_single.unsqueeze(0)
            data_fake_batch = {
                k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v)
                for k, v in data_single.items()
            }
            constraints_output = constraints_fn(data_fake_batch, y_fake_batch)
            return constraints_output.squeeze(0)

        data_in_dims = {k: 0 for k in data.keys()}
        batch_jac_fn = vmap(
            jacrev(single_constraint_fn, argnums=0), 
            in_dims=(0, data_in_dims)
        )
        B = batch_jac_fn(y, data)
        return B
        
    def forward(self, data: Dict, y_pred: torch.Tensor, 
                constraints_fn: Callable,
                needs_proj: Optional[torch.Tensor] = None
        ) -> torch.Tensor:

        batch_size, output_dim = y_pred.shape
        constraints = constraints_fn(data, y_pred)
        
        B = self.compute_batch_jacobian(data, y_pred, constraints_fn)
        B = B.view(batch_size, -1, output_dim)
        
        W_inv_diag = torch.empty(output_dim, device=y_pred.device)
        W_inv_diag[:self.n_traj].fill_(1.0 / self.w_traj)
        W_inv_diag[self.n_traj:].fill_(1.0 / self.w_slack)
        
        B_W = B * W_inv_diag  # (batch, n_constraints, output_dim)
        
        # A = J @ W^{-1} @ J^T + reg·I
        A = torch.baddbmm(
            torch.eye(B.shape[1], device=B.device, dtype=torch.float32).mul_(1e-4).unsqueeze(0),
            B_W,
            B.transpose(1, 2)
        )
        
        # Cholesky 求逆
        try:
            inv_A = torch.cholesky_inverse(torch.linalg.cholesky(A))
        except RuntimeError:
            inv_A = torch.linalg.inv(A)
        
        # correction = W^{-1} @ J^T @ A^{-1} @ h，合并 bmm
        correction = torch.bmm(
            B_W.transpose(1, 2),
            torch.bmm(inv_A, constraints.unsqueeze(-1))
        ).squeeze(-1)
        
        # if needs_proj is None:
        #     return y_pred - correction
        
        mask = needs_proj.unsqueeze(-1).to(y_pred.dtype)
        return y_pred - mask * correction
    
class AdaNP(nn.Module):
    def __init__(self, n_outputs: int, n_constraints: int, max_depth: int = 50, tol: float = 1e-3):
        super(AdaNP, self).__init__()
        self.max_depth = max_depth
        self.tol = tol
        self.projection_layer = NeuralProjection(n_traj=n_outputs, n_slack=n_constraints)
    
    def forward(self, data: Dict, y_pred: torch.Tensor, 
        constraints_fn: Callable) -> Tuple[torch.Tensor, int]:

        y_current = y_pred
        best_y = y_current.clone()
        
        best_depth = torch.zeros(y_pred.shape[0], dtype=torch.long, device=y_pred.device)
        actual_depth = torch.zeros(y_pred.shape[0], dtype=torch.long, device=y_pred.device)
        
        with torch.no_grad():
            constraints_val = constraints_fn(data, y_current)
            min_residual = constraints_val.max(dim=-1).values  # (B,)
        
        for i in range(self.max_depth):
            with torch.no_grad():
                per_sample_residual = constraints_val.max(dim=-1).values  # (B,)
                needs_proj = per_sample_residual >= self.tol  # (B,) bool tensor

            if not needs_proj.any():
                break
            
            actual_depth += needs_proj.long()
            
            y_current = self.projection_layer(data, y_current, constraints_fn, needs_proj)
            
            with torch.no_grad():
                constraints_val = constraints_fn(data, y_current)
                new_residual = constraints_val.max(dim=-1).values  # (B,)
                
                better_mask = new_residual < min_residual
                min_residual = torch.where(better_mask, new_residual, min_residual)
                best_y = torch.where(better_mask.unsqueeze(-1), y_current, best_y)
                best_depth[better_mask] = i
        
        return best_y, best_depth, actual_depth

class ENFORCE(nn.Module):
    """
    ENFORCE: 带有自适应神经投影的非线性约束学习架构
    """
    
    def __init__(self, backbone: nn.Module, num_constraints: int,
                 max_depth: int = 100, inference_tol: float = 1e-6,
                 training_tol: float = 1e-4):
        super(ENFORCE, self).__init__()
        
        self.backbone = backbone
        self.adanp = AdaNP(max_depth=max_depth, tol=inference_tol)
        self.training_tol = training_tol
        
        # 训练状态跟踪
        self.adaptive_training = True
    def forward(self, data: torch.Tensor, constraints_fn: Callable) -> Tuple[torch.Tensor, dict]:
        y_pred = self.backbone(data)
        info = {
            'projection_depth': 0,
            'constraint_residual': 0.0,
            'projection_displacement': 0.0
        }
        
        y_final, projection_depth, actual_depth = self.adanp(data, y_pred, constraints_fn)
        info['projection_depth'] = projection_depth
        info['actual_depth'] = actual_depth

        # 计算统计信息
        with torch.no_grad():
            info['constraint_residual'] = torch.max(
                constraints_fn(data, y_final)).item()
            info['projection_displacement'] = torch.mean(
                (y_final - y_pred)**2).item()
        
        return y_final, info, y_pred
 
            
@torch.compile(fullgraph=True)
def _compiled_projection_math(y_pred: torch.Tensor, h: torch.Tensor, 
                              B: torch.Tensor, B_W: torch.Tensor, reg: torch.Tensor) -> torch.Tensor:
    # A = B_W @ B^T + reg
    A = torch.addmm(reg, B_W, B.T)
    h_unsqueeze = h.unsqueeze(-1)
    
    L = torch.linalg.cholesky(A)
    x = torch.cholesky_solve(h_unsqueeze, L)
    
    correction = torch.mm(B_W.T, x).squeeze(-1)
    return y_pred - correction.unsqueeze(0)

class NeuralProjectionTest(nn.Module):
    def __init__(self, n_traj: int = 80, n_slack: int = 200,
                 w_traj: float = 5.0, w_slack: float = 1.0):
        super().__init__()
        self.n_traj = n_traj
        self.w_traj = w_traj
        self.w_slack = w_slack
        self.output_dim = n_traj + n_slack

        self._W_inv_diag: Optional[torch.Tensor] = None
        self._reg_eye: Optional[torch.Tensor] = None
        self._jac_fn: Optional[Callable] = None

    def reset_cache(self) -> None:
        self._jac_fn = None

    def _get_jac_fn(self, constraints_fn: Callable) -> Callable:
        if self._jac_fn is None:
            def fn(y: torch.Tensor, data_: Dict) -> torch.Tensor:
                return constraints_fn(data_, y).squeeze(0)  
            self._jac_fn = jacrev(fn, argnums=0)
        return self._jac_fn

    @torch.compiler.disable
    def _get_W_inv_diag(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._W_inv_diag is None or self._W_inv_diag.device != device:
            W_inv = torch.empty(self.output_dim, device=device, dtype=dtype)
            W_inv[:self.n_traj].fill_(1.0 / self.w_traj)
            W_inv[self.n_traj:].fill_(1.0 / self.w_slack)
            self._W_inv_diag = W_inv
        return self._W_inv_diag

    @torch.compiler.disable
    def _get_reg_eye(self, m: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._reg_eye is None or self._reg_eye.shape[-1] != m or self._reg_eye.device != device:
            self._reg_eye = torch.eye(m, device=device, dtype=dtype).mul_(1e-6)
        return self._reg_eye

    @torch.compiler.disable
    def forward(
        self,
        data: Dict,
        y_pred: torch.Tensor,
        constraints_fn: Callable,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        with torch.no_grad():
            h = constraints_fn(data, y_pred).squeeze(0)  # (m,)

        jac_fn = self._get_jac_fn(constraints_fn)
        J = jac_fn(y_pred, data)
        B = J.squeeze(1)  # (m, D)
        m = B.shape[0]

        B_W = B * self._get_W_inv_diag(B.device, B.dtype)
        reg = self._get_reg_eye(m, y_pred.device, y_pred.dtype)

        y_new = _compiled_projection_math(y_pred, h, B, B_W, reg)

        with torch.no_grad():
            new_constraints_val = constraints_fn(data, y_new)

        return y_new, new_constraints_val


class AdaNPTest(nn.Module):
    def __init__(self, max_depth: int = 50, tol: float = 1e-3):
        super().__init__()
        self.max_depth = max_depth
        self.tol = tol
        self.projection_layer = NeuralProjectionTest()

    def forward(
        self,
        data: Dict,
        y_pred: torch.Tensor,
        constraints_fn: Callable,
    ) -> torch.Tensor:

        y_current = y_pred

        with torch.no_grad():
            constraints_val = constraints_fn(data, y_current)
            min_residual = constraints_val.max()

        # 循环外分配好内存
        best_y = y_current.clone()
        actual_depth = 0
        for _ in range(self.max_depth):
            with torch.no_grad():
                residual = constraints_val.max()
                if residual.item() < self.tol:
                    # pass    # for ablation
                    break
                actual_depth += 1
            # torch.compiler.cudagraph_mark_step_begin()
            y_current, constraints_val = self.projection_layer(data, y_current, constraints_fn)

            with torch.no_grad():
                new_residual = constraints_val.max()
                if new_residual < min_residual:
                    min_residual = new_residual
                    best_y.copy_(y_current)

        return best_y, actual_depth