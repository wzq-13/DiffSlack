import torch
import torch.nn as nn
from torch.func import vmap, jacrev
from typing import Callable, Dict, Optional, Tuple

class ENFORCEProjection(nn.Module):
    """
    Efficient ENFORCE-style projection for inequality constraints g(p) <= 0.

    Instead of computing jacrev(phi([p, lambda])) over the full extended variable,
    this layer computes only J_g = dg/dp and constructs the FB Jacobian analytically.
    """

    def __init__(
        self,
        n_out: int = 80,
        n_constraints: int = 120,
        w_out: float = 1.0,
        w_lambda: float = 1.0,
        damping: float = 1e-4,
        eps_fb: float = 1e-12,
    ):
        super().__init__()
        self.n_out = n_out
        self.n_constraints = n_constraints
        self.n_ext = n_out + n_constraints

        self.w_out = w_out
        self.w_lambda = w_lambda
        self.damping = damping
        self.eps_fb = eps_fb

    def initialize_extended_output(self, p_pred: torch.Tensor) -> torch.Tensor:
        lam0 = torch.zeros(
            p_pred.shape[0],
            self.n_constraints,
            device=p_pred.device,
            dtype=p_pred.dtype,
        )
        return torch.cat([p_pred, lam0], dim=-1)

    def split(self, y_ext: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p = y_ext[:, :self.n_out]
        lam = y_ext[:, self.n_out:self.n_out + self.n_constraints]
        return p, lam

    def fb_residual(
        self,
        data: Dict,
        y_ext: torch.Tensor,
        constraints_fn: Callable,
    ) -> torch.Tensor:
        p, lam = self.split(y_ext)
        g = constraints_fn(data, p)  # (B, Nc), original inequality g(p) <= 0
        b = -g
        r = torch.sqrt(lam ** 2 + b ** 2 + self.eps_fb)
        phi = r - lam - b
        return phi

    def compute_batch_jg(
        self,
        data: Dict,
        p: torch.Tensor,
        constraints_fn: Callable,
    ) -> torch.Tensor:
        """
        Compute J_g = dg/dp only.

        Return:
            J_g: (B, Nc, n_out)
        """

        def single_g_fn(p_single, data_single):
            p_fake_batch = p_single.unsqueeze(0)
            data_fake_batch = {
                k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v)
                for k, v in data_single.items()
            }
            g = constraints_fn(data_fake_batch, p_fake_batch)
            return g.squeeze(0)  # (Nc,)

        data_in_dims = {k: 0 for k in data.keys()}

        batch_jac_fn = vmap(
            jacrev(single_g_fn, argnums=0),
            in_dims=(0, data_in_dims),
        )

        J_g = batch_jac_fn(p, data)
        return J_g.view(p.shape[0], self.n_constraints, self.n_out)

    def forward(
        self,
        data: Dict,
        y_ext: torch.Tensor,
        constraints_fn: Callable,
        needs_proj: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return:
            y_new: projected extended output
            phi_new: new FB residual
        """

        p, lam = self.split(y_ext)

        # Original inequality and FB residual
        g = constraints_fn(data, p)  # (B, Nc)
        b = -g
        r = torch.sqrt(lam ** 2 + b ** 2 + self.eps_fb)
        phi = r - lam - b  # (B, Nc)

        # Only compute dg/dp
        J_g = self.compute_batch_jg(data, p, constraints_fn)  # (B, Nc, n_out)

        # Analytic FB derivatives
        coef_g = 1.0 - b / r          # d phi / d g, (B, Nc)
        coef_lam = lam / r - 1.0      # d phi / d lambda, (B, Nc)

        # J_p = diag(coef_g) @ J_g
        J_p = coef_g.unsqueeze(-1) * J_g  # (B, Nc, n_out)

        w_out_inv = 1.0 / self.w_out
        w_lam_inv = 1.0 / self.w_lambda

        # A = J W^{-1} J^T + damping I
        # Since lambda block is diagonal, avoid explicitly constructing J_lam.
        Jp_W = J_p * w_out_inv

        eye = torch.eye(
            self.n_constraints,
            device=y_ext.device,
            dtype=y_ext.dtype,
        ).unsqueeze(0)

        A = torch.baddbmm(
            eye.mul(self.damping),
            Jp_W,
            J_p.transpose(1, 2),
        )

        # Add lambda diagonal block:
        # diag(coef_lam) * w_lam_inv * diag(coef_lam)
        A.diagonal(dim1=-2, dim2=-1).add_(w_lam_inv * coef_lam ** 2)

        rhs = phi.unsqueeze(-1)

        # Fast Cholesky path, same spirit as your own layer.
        # Avoid cholesky_ex + Python branch during training.
        try:
            L = torch.linalg.cholesky(A)
            alpha = torch.cholesky_solve(rhs, L).squeeze(-1)  # (B, Nc)
        except RuntimeError:
            # Fallback to pinverse for numerical stability, with warning.
            A_inv = torch.linalg.pinv(A)
            alpha = torch.bmm(A_inv, rhs).squeeze(-1)
            
        # correction_p = Wp^{-1} J_p^T alpha
        correction_p = w_out_inv * torch.bmm(
            J_p.transpose(1, 2),
            alpha.unsqueeze(-1),
        ).squeeze(-1)

        # correction_lambda = Wlambda^{-1} diag(coef_lam)^T alpha
        correction_lam = w_lam_inv * coef_lam * alpha

        correction = torch.cat([correction_p, correction_lam], dim=-1)

        if needs_proj is not None:
            mask = needs_proj.unsqueeze(-1).to(y_ext.dtype)
            y_new = y_ext - mask * correction
        else:
            y_new = y_ext - correction

        with torch.no_grad():
            phi_new = self.fb_residual(data, y_new, constraints_fn)

        return y_new, phi_new
    
class ENFORCEAadaNP(nn.Module):
    def __init__(
        self,
        n_out: int = 80,
        n_constraints: int = 120,
        max_depth: int = 50,
        tol: float = 1e-3,
        w_out: float = 1.0,
        w_lambda: float = 1.0,
        damping: float = 1e-4,
        eps_fb: float = 1e-12,
    ):
        super().__init__()
        self.n_out = n_out
        self.n_constraints = n_constraints
        self.max_depth = max_depth
        self.tol = tol

        self.projection_layer = ENFORCEProjection(
            n_out=n_out,
            n_constraints=n_constraints,
            w_out=w_out,
            w_lambda=w_lambda,
            damping=damping,
            eps_fb=eps_fb,
        )

    def forward(
        self,
        data: Dict,
        p_pred: torch.Tensor,
        constraints_fn: Callable,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:

        y_current = self.projection_layer.initialize_extended_output(p_pred)

        best_y = y_current.clone()
        actual_depth = torch.zeros(
            p_pred.shape[0],
            dtype=torch.long,
            device=p_pred.device,
        )
        best_depth = torch.zeros_like(actual_depth)

        with torch.no_grad():
            phi = self.projection_layer.fb_residual(
                data,
                y_current,
                constraints_fn,
            )
            residual = phi.abs().max(dim=-1).values
            best_residual = residual.clone()

        for i in range(self.max_depth):
            with torch.no_grad():
                needs_proj = residual >= self.tol

            if not needs_proj.any():
                break

            actual_depth += needs_proj.long()

            y_current, phi = self.projection_layer(
                data,
                y_current,
                constraints_fn,
                needs_proj=needs_proj,
            )

            with torch.no_grad():
                residual = phi.abs().max(dim=-1).values
                better_mask = residual < best_residual

                best_residual = torch.where(
                    better_mask,
                    residual,
                    best_residual,
                )

                best_y = torch.where(
                    better_mask.unsqueeze(-1),
                    y_current,
                    best_y,
                )

                best_depth[better_mask] = i + 1

        p_final = best_y[:, :self.n_out]

        with torch.no_grad():
            g_final = constraints_fn(data, p_final)
            max_ineq_violation = g_final.max(dim=-1).values

            info = {
                "fb_residual": best_residual,
                "max_ineq_violation": max_ineq_violation,
                "best_depth": best_depth,
                "actual_depth": actual_depth,
                "projection_displacement": torch.mean(
                    (p_final - p_pred) ** 2,
                    dim=-1,
                ),
            }

        return p_final, actual_depth, info
# ============================================================
# 1. ENFORCE test 版的纯数学投影核
# ============================================================

@torch.compile(fullgraph=True)
def _compiled_enforce_projection_math(
    y_ext: torch.Tensor,      # (1, D_ext)
    phi: torch.Tensor,        # (Nc,)
    J: torch.Tensor,          # (Nc, D_ext)
    J_W: torch.Tensor,        # (Nc, D_ext)
    reg: torch.Tensor,        # (Nc, Nc)
) -> torch.Tensor:
    """
    One ENFORCE-style projection step:

        y_new = y - W^{-1} J^T (J W^{-1} J^T + reg)^(-1) phi

    This function is intentionally branch-free for torch.compile(fullgraph=True).
    """
    A = torch.addmm(reg, J_W, J.T)  # (Nc, Nc)

    phi_unsqueeze = phi.unsqueeze(-1)  # (Nc, 1)

    L = torch.linalg.cholesky(A)
    alpha = torch.cholesky_solve(phi_unsqueeze, L)  # (Nc, 1)

    correction = torch.mm(J_W.T, alpha).squeeze(-1)  # (D_ext,)

    return y_ext - correction.unsqueeze(0)  # (1, D_ext)


# ============================================================
# 2. Fischer-Burmeister residual wrapper
# ============================================================

class ENFORCEFBWrapper(nn.Module):
    """
    Wrap original inequality constraints g(p) <= 0 into FB equalities.

    constraints_fn(data, p) should return:
        g(p): (1, Nc)

    y_ext = [p, lambda]:
        p:      (1, n_out)
        lambda: (1, Nc)

    returns:
        phi(lambda, -g(p)): (1, Nc)
    """

    def __init__(
        self,
        n_out: int,
        n_constraints: int,
        eps_fb: float = 1e-12,
    ):
        super().__init__()
        self.n_out = n_out
        self.n_constraints = n_constraints
        self.eps_fb = eps_fb

    def split(self, y_ext: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p = y_ext[:, :self.n_out]
        lam = y_ext[:, self.n_out:self.n_out + self.n_constraints]
        return p, lam

    def forward(
        self,
        data: Dict,
        y_ext: torch.Tensor,
        constraints_fn: Callable,
    ) -> torch.Tensor:
        p, lam = self.split(y_ext)

        # 原始不等式约束: g(p) <= 0
        g = constraints_fn(data, p)  # (1, Nc)

        # ENFORCE FB reformulation uses b = -g >= 0
        b = -g

        # phi(lambda, b) = sqrt(lambda^2 + b^2 + eps) - lambda - b
        phi = torch.sqrt(lam ** 2 + b ** 2 + self.eps_fb) - lam - b

        return phi  # (1, Nc)

    @torch.no_grad()
    def max_original_violation(
        self,
        data: Dict,
        y_ext: torch.Tensor,
        constraints_fn: Callable,
    ) -> torch.Tensor:
        p, _ = self.split(y_ext)
        g = constraints_fn(data, p)
        return g.max()


# ============================================================
# 3. Batch size = 1 的 ENFORCE projection layer
# ============================================================

class ENFORCENeuralProjectionTest(nn.Module):
    """
    Test-time optimized ENFORCE projection layer for batch size = 1.

    It projects in the extended space:

        y_ext = [p, lambda]

    using FB equality residuals.
    """

    def __init__(
        self,
        n_out: int = 80,
        n_constraints: int = 120,
        w_out: float = 1.0,
        w_lambda: float = 1.0,
        damping: float = 1e-6,
        eps_fb: float = 1e-12,
    ):
        super().__init__()

        self.n_out = n_out
        self.n_constraints = n_constraints
        self.n_ext = n_out + n_constraints

        self.w_out = w_out
        self.w_lambda = w_lambda
        self.damping = damping

        self.fb = ENFORCEFBWrapper(
            n_out=n_out,
            n_constraints=n_constraints,
            eps_fb=eps_fb,
        )

        self._W_inv_diag: Optional[torch.Tensor] = None
        self._reg_eye: Optional[torch.Tensor] = None
        self._jac_fn: Optional[Callable] = None

    def reset_cache(self) -> None:
        self._jac_fn = None
        self._W_inv_diag = None
        self._reg_eye = None

    def initialize_extended_output(self, p_pred: torch.Tensor) -> torch.Tensor:
        """
        ENFORCE-style initialization:
        backbone predicts p only, and lambda is initialized to zero.
        """
        lam0 = torch.zeros(
            p_pred.shape[0],
            self.n_constraints,
            device=p_pred.device,
            dtype=p_pred.dtype,
        )
        return torch.cat([p_pred, lam0], dim=-1)  # (1, n_out + Nc)

    def _get_jac_fn(self, constraints_fn: Callable) -> Callable:
        """
        Cache jacrev function.

        Important:
        If you change constraints_fn object, call reset_cache().
        """
        if self._jac_fn is None:

            def fn(y_ext: torch.Tensor, data_: Dict) -> torch.Tensor:
                phi = self.fb(data_, y_ext, constraints_fn)
                return phi.squeeze(0)  # (Nc,)

            self._jac_fn = jacrev(fn, argnums=0)

        return self._jac_fn

    @torch.compiler.disable
    def _get_W_inv_diag(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if (
            self._W_inv_diag is None
            or self._W_inv_diag.device != device
            or self._W_inv_diag.dtype != dtype
        ):
            W_inv = torch.empty(self.n_ext, device=device, dtype=dtype)
            W_inv[:self.n_out].fill_(1.0 / self.w_out)
            W_inv[self.n_out:].fill_(1.0 / self.w_lambda)
            self._W_inv_diag = W_inv

        return self._W_inv_diag

    @torch.compiler.disable
    def _get_reg_eye(
        self,
        m: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if (
            self._reg_eye is None
            or self._reg_eye.shape[-1] != m
            or self._reg_eye.device != device
            or self._reg_eye.dtype != dtype
        ):
            self._reg_eye = torch.eye(
                m,
                device=device,
                dtype=dtype,
            ).mul_(self.damping)

        return self._reg_eye

    @torch.compiler.disable
    def forward(
        self,
        data: Dict,
        y_ext: torch.Tensor,
        constraints_fn: Callable,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            data: input dict
            y_ext: (1, n_out + n_constraints)
            constraints_fn: original inequality function g(p) <= 0

        Returns:
            y_new: new extended output
            phi_new: new FB residual, shape (1, Nc)
            g_new: original inequality value, shape (1, Nc)
        """

        # Current FB residual
        with torch.no_grad():
            phi = self.fb(data, y_ext, constraints_fn).squeeze(0)  # (Nc,)

        # Jacobian of FB residual wrt y_ext
        jac_fn = self._get_jac_fn(constraints_fn)
        J = jac_fn(y_ext, data)  # (Nc, 1, D_ext)
        J = J.squeeze(1)         # (Nc, D_ext)

        # Weighted Jacobian: J W^{-1}
        W_inv_diag = self._get_W_inv_diag(J.device, J.dtype)
        J_W = J * W_inv_diag  # (Nc, D_ext)

        reg = self._get_reg_eye(
            m=J.shape[0],
            device=y_ext.device,
            dtype=y_ext.dtype,
        )

        y_new = _compiled_enforce_projection_math(
            y_ext,
            phi,
            J,
            J_W,
            reg,
        )

        with torch.no_grad():
            phi_new = self.fb(data, y_new, constraints_fn)  # (1, Nc)
            p_new = y_new[:, :self.n_out]
            g_new = constraints_fn(data, p_new)             # (1, Nc)

        return y_new, phi_new, g_new


# ============================================================
# 4. Batch size = 1 的 ENFORCE AdaNP test 版
# ============================================================

class ENFORCEAadaNPTest(nn.Module):
    """
    Test-time optimized ENFORCE AdaNP for batch size = 1.

    The stopping condition uses FB residual:
        max |phi| < tol

    Final feasibility should be evaluated using original inequality:
        max g(p) <= 0
    """

    def __init__(
        self,
        n_out: int = 80,
        n_constraints: int = 120,
        max_depth: int = 50,
        tol: float = 1e-3,
        w_out: float = 1.0,
        w_lambda: float = 1.0,
        damping: float = 1e-6,
        eps_fb: float = 1e-12,
    ):
        super().__init__()

        self.n_out = n_out
        self.n_constraints = n_constraints
        self.max_depth = max_depth
        self.tol = tol

        self.projection_layer = ENFORCENeuralProjectionTest(
            n_out=n_out,
            n_constraints=n_constraints,
            w_out=w_out,
            w_lambda=w_lambda,
            damping=damping,
            eps_fb=eps_fb,
        )

    def forward(
        self,
        data: Dict,
        p_pred: torch.Tensor,
        constraints_fn: Callable,
    ) -> Tuple[torch.Tensor, int, Dict]:
        """
        Args:
            p_pred: (1, n_out), raw trajectory prediction
            constraints_fn: original inequality function g(p) <= 0

        Returns:
            p_best: (1, n_out)
            actual_depth: int
            info: dict
        """

        y_current = self.projection_layer.initialize_extended_output(p_pred)

        with torch.no_grad():
            phi = self.projection_layer.fb(data, y_current, constraints_fn)
            residual = phi.abs().max()

            p_current = y_current[:, :self.n_out]
            g_current = constraints_fn(data, p_current)
            max_ineq_violation = g_current.max()

            min_residual = residual

        best_y = y_current.clone()
        actual_depth = 0

        for _ in range(self.max_depth):
            with torch.no_grad():
                # ENFORCE 的迭代标准应该看 FB equality residual
                residual = phi.abs().max()

                # 这里和你的版本一样，有一次 CUDA 同步用于 early stop
                if residual.item() < self.tol:
                    break

                actual_depth += 1

            y_current, phi, g_current = self.projection_layer(
                data,
                y_current,
                constraints_fn,
            )

            with torch.no_grad():
                new_residual = phi.abs().max()

                if new_residual < min_residual:
                    min_residual = new_residual
                    best_y.copy_(y_current)

        p_best = best_y[:, :self.n_out]

        with torch.no_grad():
            g_best = constraints_fn(data, p_best)
            final_ineq_violation = g_best.max()
            final_fb_residual = self.projection_layer.fb(
                data,
                best_y,
                constraints_fn,
            ).abs().max()

            info = {
                "actual_depth": actual_depth,
                "fb_residual": final_fb_residual.item(),
                "max_ineq_violation": final_ineq_violation.item(),
                "projection_displacement": torch.mean(
                    (p_best - p_pred) ** 2
                ).item(),
            }

        return p_best, actual_depth, info