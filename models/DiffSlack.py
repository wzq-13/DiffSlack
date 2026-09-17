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

SLACK_INITIALIZATION_MODES = (
    "learned", "zero", "constant", "analytic",
)


def validate_slack_initialization(mode: str) -> str:
    if mode not in SLACK_INITIALIZATION_MODES:
        choices = ", ".join(SLACK_INITIALIZATION_MODES)
        raise ValueError(f"Unknown slack initialization '{mode}'; choose from {choices}")
    return mode


def _nonnegative_sqrt(value: torch.Tensor) -> torch.Tensor:
    """sqrt(ReLU(value)) with a finite zero derivative at value <= 0."""
    positive = value > 0
    safe = torch.where(positive, value, torch.ones_like(value))
    return torch.where(positive, safe.sqrt(), torch.zeros_like(value))

RESULT_DIR = './test_hard2_logs'
CSV_FILE_PATH = os.path.join(RESULT_DIR, 'batch_details.csv')
SUMMARY_FILE_PATH = os.path.join(RESULT_DIR, 'final_summary.txt')


def _interleaved_inverse_weights(n_traj: int, n_slack: int,
                                 w_traj: float, w_slack: float,
                                 device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build W^-1 for waypoint rows laid out as [x, y, slack_1, ..., slack_k]."""
    if n_traj % 2 != 0:
        raise ValueError(f"n_traj must contain x/y pairs, got {n_traj}")
    horizon = n_traj // 2
    if horizon == 0 or n_slack % horizon != 0:
        raise ValueError(
            f"n_slack={n_slack} must be divisible by the horizon {horizon}"
        )

    slack_per_waypoint = n_slack // horizon
    weights = torch.empty(
        horizon, 2 + slack_per_waypoint, device=device, dtype=dtype
    )
    weights[:, :2].fill_(1.0 / w_traj)
    weights[:, 2:].fill_(1.0 / w_slack)
    return weights.flatten()


class NeuralProjection(nn.Module):
    def __init__(self, n_traj=80, n_slack=200, w_traj=5.0, w_slack=1.0,
                 damping=1e-4):
        super().__init__()
        self.n_traj = n_traj
        self.n_slack = n_slack
        self.w_traj = w_traj
        self.w_slack = w_slack
        self.damping = damping
        
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
        
        W_inv_diag = _interleaved_inverse_weights(
            self.n_traj, self.n_slack, self.w_traj, self.w_slack,
            y_pred.device, y_pred.dtype,
        )
        if W_inv_diag.numel() != output_dim:
            raise ValueError(
                f"Expected interleaved output dimension {W_inv_diag.numel()}, got {output_dim}"
            )
        
        B_W = B * W_inv_diag  # (batch, n_constraints, output_dim)
        
        # A = J @ W^{-1} @ J^T + reg·I
        A = torch.baddbmm(
            torch.eye(B.shape[1], device=B.device, dtype=B.dtype).mul_(self.damping).unsqueeze(0),
            B_W,
            B.transpose(1, 2)
        )
        
        # Compute the inverse via Cholesky factorization
        try:
            inv_A = torch.cholesky_inverse(torch.linalg.cholesky(A))
        except RuntimeError:
            inv_A = torch.linalg.inv(A)
        
        # correction = W^{-1} @ J^T @ A^{-1} @ h; combine the bmm operations
        correction = torch.bmm(
            B_W.transpose(1, 2),
            torch.bmm(inv_A, constraints.unsqueeze(-1))
        ).squeeze(-1)
        
        # if needs_proj is None:
        #     return y_pred - correction
        
        mask = needs_proj.unsqueeze(-1).to(y_pred.dtype)
        return y_pred - mask * correction
    
class AdaNP(nn.Module):
    def __init__(self, n_outputs: int, n_constraints: int, max_depth: int = 50,
                 tol: float = 1e-3,
                 initialization_mode: str = "learned",
                 initialization_constant: float = 0.1,
                 w_traj: float = 5.0, w_slack: float = 1.0,
                 damping: float = 1e-4):
        super(AdaNP, self).__init__()
        self.max_depth = max_depth
        self.tol = tol
        self.initialization_mode = validate_slack_initialization(initialization_mode)
        self.initialization_constant = float(initialization_constant)
        if not np.isfinite(self.initialization_constant):
            raise ValueError("initialization_constant must be finite")
        self.projection_layer = NeuralProjection(
            n_traj=n_outputs,
            n_slack=n_constraints,
            w_traj=w_traj,
            w_slack=w_slack,
            damping=damping,
        )

    def initialize_output(self, data: Dict, y_pred: torch.Tensor,
                          constraints_fn: Callable) -> torch.Tensor:
        """Keep the predicted path and select only the squared-slack initialization."""
        if self.initialization_mode == "learned":
            return y_pred

        batch_size = y_pred.shape[0]
        horizon = self.projection_layer.n_traj // 2
        slack_per_waypoint = self.projection_layer.n_slack // horizon
        expected_dim = horizon * (2 + slack_per_waypoint)
        if slack_per_waypoint != 5 or y_pred.shape[1] != expected_dim:
            raise ValueError(
                "DiffSlack initialization expects five slack variables per waypoint"
            )
        path = y_pred.view(batch_size, horizon, 7)[:, :, :2]

        if self.initialization_mode == "zero":
            slack = y_pred.new_zeros(batch_size, horizon, 5)
        elif self.initialization_mode == "constant":
            slack = y_pred.new_full(
                (batch_size, horizon, 5), self.initialization_constant
            )
        else:
            # Evaluate g(p) by setting every slack to zero.  The collision row is
            # the same three-circle LSE used by the main DiffSlack residual.  A
            # common value for its three slacks shifts that LSE by s^2 exactly.
            with torch.no_grad():
                zero_slack = y_pred.new_zeros(batch_size, horizon, 5)
                zero_output = torch.cat((path.detach(), zero_slack), dim=-1).flatten(1)
                inequality = constraints_fn(data, zero_output)
                if inequality.shape[1] != 3 * horizon:
                    raise ValueError(
                        f"Analytic initialization expected {3 * horizon} residuals, "
                        f"got {inequality.shape[1]}"
                    )
                collision, curvature, distance = inequality.split(horizon, dim=-1)
                collision_slack = _nonnegative_sqrt(-collision).unsqueeze(-1).expand(
                    -1, -1, 3
                )
                slack = torch.cat((
                    collision_slack,
                    _nonnegative_sqrt(-curvature).unsqueeze(-1),
                    _nonnegative_sqrt(-distance).unsqueeze(-1),
                ), dim=-1)

        return torch.cat((path, slack), dim=-1).flatten(1)
    
    def forward(self, data: Dict, y_pred: torch.Tensor, 
        constraints_fn: Callable) -> Tuple[torch.Tensor, int]:

        y_pred = self.initialize_output(data, y_pred, constraints_fn)

        best_y, best_depth, actual_depth = self._iterations(
            data, y_pred, constraints_fn
        )
        return best_y, best_depth, actual_depth

    def _iterations(self, data, y_pred, constraints_fn):

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
                best_depth[better_mask] = i
                best_y = torch.where(better_mask.unsqueeze(-1), y_current, best_y)
        
        return best_y, best_depth, actual_depth

class DiffSlack(nn.Module):
    
    def __init__(self, backbone: nn.Module, num_constraints: int,
                 max_depth: int = 100, inference_tol: float = 1e-6,
                 training_tol: float = 1e-4,
                 initialization_mode: str = "learned",
                 initialization_constant: float = 0.1):
        super(DiffSlack, self).__init__()
        
        self.backbone = backbone
        self.adanp = AdaNP(
            n_outputs=80, n_constraints=num_constraints,
            max_depth=max_depth, tol=inference_tol,
            initialization_mode=initialization_mode,
            initialization_constant=initialization_constant,
        )
        self.training_tol = training_tol
        
        # Track training state
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

        # Compute statistics
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
                 w_traj: float = 5.0, w_slack: float = 1.0,
                 damping: float = 1e-4):
        super().__init__()
        self.n_traj = n_traj
        self.w_traj = w_traj
        self.w_slack = w_slack
        self.damping = damping
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
        if (self._W_inv_diag is None or self._W_inv_diag.device != device
                or self._W_inv_diag.dtype != dtype):
            self._W_inv_diag = _interleaved_inverse_weights(
                self.n_traj, self.output_dim - self.n_traj,
                self.w_traj, self.w_slack, device, dtype,
            )
        return self._W_inv_diag

    @torch.compiler.disable
    def _get_reg_eye(self, m: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if (
            self._reg_eye is None
            or self._reg_eye.shape[-1] != m
            or self._reg_eye.device != device
            or self._reg_eye.dtype != dtype
        ):
            self._reg_eye = torch.eye(
                m, device=device, dtype=dtype
            ).mul_(self.damping)
        return self._reg_eye

    @torch.compiler.disable
    def forward(
        self,
        data: Dict,
        y_pred: torch.Tensor,
        constraints_fn: Callable,
        profile: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        with torch.no_grad():
            h = constraints_fn(data, y_pred).squeeze(0)  # (m,)

        jac_fn = self._get_jac_fn(constraints_fn)
        if profile and y_pred.is_cuda:
            torch.cuda.synchronize(y_pred.device)
        jacobian_start = time.perf_counter()
        J = jac_fn(y_pred, data)
        if profile and y_pred.is_cuda:
            torch.cuda.synchronize(y_pred.device)
        jacobian_time = time.perf_counter() - jacobian_start
        B = J.squeeze(1)  # (m, D)
        m = B.shape[0]

        B_W = B * self._get_W_inv_diag(B.device, B.dtype)
        reg = self._get_reg_eye(m, y_pred.device, y_pred.dtype)

        if profile:
            A = torch.addmm(reg, B_W, B.T)
            if y_pred.is_cuda:
                torch.cuda.synchronize(y_pred.device)
            cholesky_start = time.perf_counter()
            try:
                L = torch.linalg.cholesky(A)
                inv_A = None
            except RuntimeError:
                # Match the training projection: one Cholesky attempt followed
                # by a direct inverse fallback.  The fallback cost remains in
                # this factorization timing bucket.
                L = None
                inv_A = torch.linalg.inv(A)
            if y_pred.is_cuda:
                torch.cuda.synchronize(y_pred.device)
            cholesky_time = time.perf_counter() - cholesky_start

            solve_start = time.perf_counter()
            if L is not None:
                x = torch.cholesky_solve(h.unsqueeze(-1), L)
            else:
                x = torch.mm(inv_A, h.unsqueeze(-1))
            correction = torch.mm(B_W.T, x).squeeze(-1)
            y_new = y_pred - correction.unsqueeze(0)
            if y_pred.is_cuda:
                torch.cuda.synchronize(y_pred.device)
            solve_time = time.perf_counter() - solve_start
        else:
            try:
                y_new = _compiled_projection_math(y_pred, h, B, B_W, reg)
            except RuntimeError:
                # Same single-attempt fallback used during training.
                A = torch.addmm(reg, B_W, B.T)
                inv_A = torch.linalg.inv(A)
                correction = torch.mm(
                    B_W.T, torch.mm(inv_A, h.unsqueeze(-1))
                ).squeeze(-1)
                y_new = y_pred - correction.unsqueeze(0)

        with torch.no_grad():
            new_constraints_val = constraints_fn(data, y_new)

        if profile:
            return y_new, new_constraints_val, {
                "jacobian_time": jacobian_time,
                "cholesky_time": cholesky_time,
                "linear_solve_time": solve_time,
            }
        return y_new, new_constraints_val


class AdaNPTest(nn.Module):
    def __init__(self, max_depth: int = 50, tol: float = 1e-3,
                 w_traj: float = 5.0, w_slack: float = 1.0,
                 damping: float = 1e-4):
        super().__init__()
        self.max_depth = max_depth
        self.tol = tol
        self.projection_layer = NeuralProjectionTest(
            w_traj=w_traj,
            w_slack=w_slack,
            damping=damping,
        )

    def forward(
        self,
        data: Dict,
        y_pred: torch.Tensor,
        constraints_fn: Callable,
        profile: bool = False,
    ) -> torch.Tensor:

        y_current = y_pred

        with torch.no_grad():
            constraints_val = constraints_fn(data, y_current)
            min_residual = constraints_val.max()

        # Preallocate memory outside the loop
        best_y = y_current.clone()
        actual_depth = 0
        timing = {
            "jacobian_time": 0.0,
            "cholesky_time": 0.0,
            "linear_solve_time": 0.0,
        }
        for _ in range(self.max_depth):
            with torch.no_grad():
                residual = constraints_val.max()
                if residual.item() < self.tol:
                    # pass    # for ablation
                    break
                actual_depth += 1
            # torch.compiler.cudagraph_mark_step_begin()
            projection_result = self.projection_layer(
                data, y_current, constraints_fn, profile=profile
            )
            if profile:
                y_current, constraints_val, step_timing = projection_result
                for key in timing:
                    timing[key] += step_timing[key]
            else:
                y_current, constraints_val = projection_result

            with torch.no_grad():
                new_residual = constraints_val.max()
                if new_residual < min_residual:
                    min_residual = new_residual
                    best_y.copy_(y_current)

        if profile:
            return best_y, actual_depth, timing
        return best_y, actual_depth
