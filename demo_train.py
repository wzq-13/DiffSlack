"""Train four small models on a nonlinear constrained regression problem.

The input is x in [0, 6] and the desired unconstrained response is

    f(x) = 0.65 sin(x) + 0.20 log(1 + x).

The prediction y must satisfy

    g1(x, y) = exp(y) - (1.85 + 0.32 sin(1.35 x)) <= 0,
    g2(x, y) = 0.08 + 0.18 cos(1.15 x) - log(y + 1.45) <= 0.

The exact pointwise constrained optimum for squared error to f(x) is the
nominal response clipped to the corresponding feasible interval.  This script
compares an unconstrained MLP, a soft-penalty MLP, ENFORCE, and DiffSlack.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.func import jacrev, vmap
from torch.utils.data import DataLoader, TensorDataset


METHODS = ("MLP", "Soft penalty", "ENFORCE", "DiffSlack")


def nominal_target(x: torch.Tensor) -> torch.Tensor:
    return 0.65 * torch.sin(x) + 0.20 * torch.log1p(x)


def feasible_bounds(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    upper = torch.log(1.85 + 0.32 * torch.sin(1.35 * x))
    lower = torch.exp(0.08 + 0.18 * torch.cos(1.15 * x)) - 1.45
    return lower, upper


def exact_constrained_optimum(x: torch.Tensor) -> torch.Tensor:
    lower, upper = feasible_bounds(x)
    return torch.minimum(torch.maximum(nominal_target(x), lower), upper)


def inequalities(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return [g1, g2], with feasibility defined by both values <= 0."""
    safe_log_argument = (y + 1.45).clamp_min(1e-8)
    g1 = torch.exp(y) - (1.85 + 0.32 * torch.sin(1.35 * x))
    g2 = 0.08 + 0.18 * torch.cos(1.15 * x) - torch.log(safe_log_argument)
    return torch.cat((g1, g2), dim=-1)


def inequality_derivative(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Analytical dg/dy for the two scalar-output constraints."""
    del x
    safe_log_argument = (y + 1.45).clamp_min(1e-8)
    return torch.cat((torch.exp(y), -1.0 / safe_log_argument), dim=-1)


def diffslack_jacobian(
    x: torch.Tensor, state: torch.Tensor, mode: str
) -> torch.Tensor:
    """Return d(g+s^2)/d[y,s1,s2] using the requested implementation."""
    if mode == "analytic":
        y, slack = state[:, :1], state[:, 1:]
        jacobian = state.new_zeros(state.shape[0], 2, 3)
        jacobian[:, :, 0] = inequality_derivative(x, y)
        jacobian[:, 0, 1] = 2.0 * slack[:, 0]
        jacobian[:, 1, 2] = 2.0 * slack[:, 1]
        return jacobian

    def residual_single(z: torch.Tensor, x_single: torch.Tensor) -> torch.Tensor:
        y = z[0]
        g1 = torch.exp(y) - (1.85 + 0.32 * torch.sin(1.35 * x_single[0]))
        safe_log_argument = torch.clamp_min(y + 1.45, 1e-8)
        g2 = (
            0.08 + 0.18 * torch.cos(1.15 * x_single[0])
            - torch.log(safe_log_argument)
        )
        return torch.stack((g1 + z[1].square(), g2 + z[2].square()))

    return vmap(jacrev(residual_single, argnums=0))(state, x)


def enforce_residual_and_jacobian(
    x: torch.Tensor, state: torch.Tensor, eps_fb: float, mode: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the FB residual and its analytic or autodiff Jacobian."""
    y, multiplier = state[:, :1], state[:, 1:]
    g = inequalities(x, y)
    b = -g
    radius = torch.sqrt(multiplier.square() + b.square() + eps_fb)
    phi = radius - multiplier - b

    if mode == "analytic":
        coefficient_g = 1.0 - b / radius
        coefficient_lam = multiplier / radius - 1.0
        jacobian = state.new_zeros(state.shape[0], 2, 3)
        jacobian[:, :, 0] = coefficient_g * inequality_derivative(x, y)
        jacobian[:, 0, 1] = coefficient_lam[:, 0]
        jacobian[:, 1, 2] = coefficient_lam[:, 1]
        return phi, jacobian

    def residual_single(z: torch.Tensor, x_single: torch.Tensor) -> torch.Tensor:
        y_single = z[0]
        g1 = torch.exp(y_single) - (
            1.85 + 0.32 * torch.sin(1.35 * x_single[0])
        )
        safe_log_argument = torch.clamp_min(y_single + 1.45, 1e-8)
        g2 = (
            0.08 + 0.18 * torch.cos(1.15 * x_single[0])
            - torch.log(safe_log_argument)
        )
        g_single = torch.stack((g1, g2))
        b_single = -g_single
        radius_single = torch.sqrt(z[1:].square() + b_single.square() + eps_fb)
        return radius_single - z[1:] - b_single

    jacobian = vmap(jacrev(residual_single, argnums=0))(state, x)
    return phi, jacobian


class TinyNetwork(nn.Module):
    """One-hidden-layer fully connected network with 64 hidden neurons."""

    def __init__(self, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fixed input normalization shared by all four methods.
        return self.network(x / 3.0 - 1.0)


def _batched_linear_solve(A: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Use the same one-attempt Cholesky/inverse policy as the main code."""
    try:
        L = torch.linalg.cholesky(A)
        return torch.cholesky_solve(rhs, L)
    except RuntimeError:
        return torch.bmm(torch.linalg.inv(A), rhs)


@torch.no_grad()
def project_diffslack(
    x: torch.Tensor,
    z_initial: torch.Tensor,
    max_iterations: int,
    tolerance: float,
    damping: float,
    w_y: float,
    w_slack: float,
    jacobian_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adaptive-depth Gauss-Newton projection of z=[y,s1,s2]."""
    z = z_initial.clone()
    best = z.clone()
    residual = inequalities(x, z[:, :1]) + z[:, 1:].square()
    score = residual.max(dim=-1).values
    best_score = score.clone()
    depth = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
    inverse_weights = z.new_tensor([1.0 / w_y, 1.0 / w_slack, 1.0 / w_slack])
    eye = torch.eye(2, dtype=z.dtype, device=z.device).unsqueeze(0)

    for _ in range(max_iterations):
        active = score >= tolerance
        if not active.any():
            break
        depth += active.long()

        jacobian = diffslack_jacobian(x, z, jacobian_mode)
        weighted_jacobian = jacobian * inverse_weights
        A = torch.baddbmm(
            eye.expand(z.shape[0], -1, -1) * damping,
            weighted_jacobian,
            jacobian.transpose(1, 2),
        )
        alpha = _batched_linear_solve(A, residual.unsqueeze(-1))
        correction = torch.bmm(
            weighted_jacobian.transpose(1, 2), alpha
        ).squeeze(-1)
        z_candidate = z - correction
        z = torch.where(active.unsqueeze(-1), z_candidate, z)

        residual = inequalities(x, z[:, :1]) + z[:, 1:].square()
        score = residual.max(dim=-1).values
        improved = score < best_score
        best_score = torch.where(improved, score, best_score)
        best = torch.where(improved.unsqueeze(-1), z, best)

    return best, depth


@torch.no_grad()
def project_enforce(
    x: torch.Tensor,
    y_initial: torch.Tensor,
    max_iterations: int,
    tolerance: float,
    damping: float,
    eps_fb: float,
    jacobian_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ENFORCE projection with two FB multipliers initialized to zero."""
    lam = torch.zeros(y_initial.shape[0], 2, device=y_initial.device, dtype=y_initial.dtype)
    extended = torch.cat((y_initial, lam), dim=-1)
    best = extended.clone()
    depth = torch.zeros(y_initial.shape[0], dtype=torch.long, device=y_initial.device)
    eye = torch.eye(2, dtype=y_initial.dtype, device=y_initial.device).unsqueeze(0)

    phi, jacobian = enforce_residual_and_jacobian(
        x, extended, eps_fb, jacobian_mode
    )
    score = phi.abs().max(dim=-1).values
    best_score = score.clone()
    inverse_weights = extended.new_ones(3)

    for _ in range(max_iterations):
        active = score >= tolerance
        if not active.any():
            break
        depth += active.long()
        weighted_jacobian = jacobian * inverse_weights
        A = torch.baddbmm(
            eye.expand(extended.shape[0], -1, -1) * damping,
            weighted_jacobian,
            jacobian.transpose(1, 2),
        )
        alpha = _batched_linear_solve(A, phi.unsqueeze(-1))
        correction = torch.bmm(
            weighted_jacobian.transpose(1, 2), alpha
        ).squeeze(-1)
        candidate = extended - correction
        extended = torch.where(active.unsqueeze(-1), candidate, extended)
        phi, jacobian = enforce_residual_and_jacobian(
            x, extended, eps_fb, jacobian_mode
        )
        score = phi.abs().max(dim=-1).values
        improved = score < best_score
        best_score = torch.where(improved, score, best_score)
        best = torch.where(improved.unsqueeze(-1), extended, best)

    return best[:, :1], depth


def soft_violation_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.relu(inequalities(x, y)).square().mean()


def make_training_data(
    seed: int, n_train: int, noise_std: float, dtype: torch.dtype
):
    generator = torch.Generator().manual_seed(seed)
    # Generate a shared float64 dataset so float32/float64 comparisons use
    # identical input points and noise realizations.
    x64 = torch.rand(n_train, 1, generator=generator, dtype=torch.float64) * 6.0
    noise64 = (
        torch.randn(n_train, 1, generator=generator, dtype=torch.float64)
        * noise_std
    )
    labels64 = nominal_target(x64) + noise64
    return x64.to(dtype), labels64.to(dtype)


def train_method(
    method: str,
    seed: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> nn.Module:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    output_dim = 3 if method == "DiffSlack" else 1
    model = TinyNetwork(output_dim, args.hidden_dim).to(
        device=device, dtype=x_train.dtype
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    dataset = TensorDataset(x_train, y_train)
    loader_generator = torch.Generator().manual_seed(seed + 1000)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    total_epochs = args.stage1_epochs + args.stage2_epochs
    method_penalty_override = {
        "Soft penalty": getattr(args, "soft_penalty_weight", None),
        "ENFORCE": getattr(args, "enforce_penalty_weight", None),
        "DiffSlack": getattr(args, "diffslack_penalty_weight", None),
    }.get(method)
    penalty_weight = (
        args.penalty_weight
        if method_penalty_override is None
        else method_penalty_override
    )

    for epoch in range(total_epochs):
        model.train()
        running_loss = 0.0
        for x_batch, label_batch in loader:
            x_batch = x_batch.to(device)
            label_batch = label_batch.to(device)
            output = model(x_batch)
            y_raw = output[:, :1]
            supervised_loss = torch.mean((y_raw - label_batch).square())

            if method == "MLP":
                loss = supervised_loss
            elif method == "Soft penalty":
                loss = supervised_loss + 50 * soft_violation_loss(
                    x_batch, y_raw
                )
            elif method == "ENFORCE":
                constraint_loss = soft_violation_loss(x_batch, y_raw)
                if epoch < args.stage1_epochs:
                    loss = supervised_loss + penalty_weight * constraint_loss
                else:
                    y_projected, _ = project_enforce(
                        x_batch, y_raw.detach(), args.max_iterations,
                        args.tolerance, args.damping, args.fb_epsilon,
                        args.jacobian_mode,
                    )
                    projection_loss = torch.mean(
                        (y_raw - y_projected.detach()).square()
                    )
                    loss = (
                        supervised_loss
                        + penalty_weight * constraint_loss
                        + args.projection_weight * projection_loss
                    )
            else:
                slack = output[:, 1:]
                equality_residual = inequalities(x_batch, y_raw) + slack.square()
                equality_loss = equality_residual.square().mean()
                if epoch < args.stage1_epochs:
                    loss = supervised_loss + penalty_weight * equality_loss
                else:
                    projected, _ = project_diffslack(
                        x_batch, output.detach(), args.max_iterations,
                        args.tolerance, args.damping,
                        args.path_weight, args.slack_weight,
                        args.jacobian_mode,
                    )
                    projection_loss = torch.mean(
                        (output - projected.detach()).square()
                    )
                    loss = (
                        supervised_loss
                        + penalty_weight * equality_loss
                        + args.projection_weight * projection_loss
                    )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            running_loss += loss.item() * x_batch.shape[0]

        if args.verbose and (
            epoch == 0 or (epoch + 1) % args.log_interval == 0
            or epoch + 1 == total_epochs
        ):
            print(
                f"seed={seed} method={method:<12} "
                f"epoch={epoch + 1:4d}/{total_epochs} "
                f"loss={running_loss / len(dataset):.6f}"
            )

    return model


@torch.no_grad()
def predict_method(
    method: str,
    model: nn.Module,
    x: torch.Tensor,
    args: argparse.Namespace,
    jacobian_mode: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    jacobian_mode = jacobian_mode or args.jacobian_mode
    output = model(x)
    if method == "DiffSlack":
        projected, depth = project_diffslack(
            x, output, args.max_iterations, args.tolerance, args.damping,
            args.path_weight, args.slack_weight,
            jacobian_mode,
        )
        return projected[:, :1], depth
    if method == "ENFORCE":
        return project_enforce(
            x, output[:, :1], args.max_iterations, args.tolerance,
            args.damping, args.fb_epsilon,
            jacobian_mode,
        )
    return output[:, :1], torch.zeros(
        x.shape[0], dtype=torch.long, device=x.device
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class Result:
    seed: int
    method: str
    rmse: float
    violation_rate: float
    maximum_violation: float
    mean_violation: float
    ct_ms: float
    ct_std_ms: float
    mean_iterations: float


@dataclass
class JacobianTiming:
    seed: int
    method: str
    autodiff_ct_ms: float
    autodiff_ct_std_ms: float
    analytic_ct_ms: float
    analytic_ct_std_ms: float
    analytic_speedup: float


@torch.no_grad()
def measure_ct(
    method: str,
    model: nn.Module,
    timing_x: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    jacobian_mode: str,
) -> Tuple[float, float]:
    for i in range(min(args.warmup_samples, timing_x.shape[0])):
        predict_method(
            method, model, timing_x[i:i + 1], args, jacobian_mode
        )
    synchronize(device)

    elapsed_ms = []
    for i in range(timing_x.shape[0]):
        synchronize(device)
        start = time.perf_counter()
        predict_method(
            method, model, timing_x[i:i + 1], args, jacobian_mode
        )
        synchronize(device)
        elapsed_ms.append((time.perf_counter() - start) * 1000.0)
    return (
        float(np.mean(elapsed_ms)),
        float(np.std(elapsed_ms, ddof=1)) if len(elapsed_ms) > 1 else 0.0,
    )


@torch.no_grad()
def evaluate_method(
    method: str,
    model: nn.Module,
    x_test: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> Tuple[Result, np.ndarray, Optional[JacobianTiming]]:
    model.eval()
    prediction, depth = predict_method(method, model, x_test, args)
    exact = exact_constrained_optimum(x_test)
    rmse = torch.sqrt(torch.mean((prediction - exact).square())).item()
    per_sample_violation = torch.relu(
        inequalities(x_test, prediction)
    ).max(dim=-1).values
    violation_rate = (per_sample_violation > args.violation_tolerance).float().mean().item()
    maximum_violation = per_sample_violation.max().item()
    mean_violation = per_sample_violation.mean().item()

    timing_x = x_test[: min(args.ct_samples, x_test.shape[0])]
    ct_ms, ct_std_ms = measure_ct(
        method, model, timing_x, args, device, args.jacobian_mode
    )

    jacobian_timing = None
    if args.compare_jacobian_ct and method in ("ENFORCE", "DiffSlack"):
        if args.jacobian_mode == "analytic":
            analytic_ct = (ct_ms, ct_std_ms)
            autodiff_ct = measure_ct(
                method, model, timing_x, args, device, "autodiff"
            )
        else:
            autodiff_ct = (ct_ms, ct_std_ms)
            analytic_ct = measure_ct(
                method, model, timing_x, args, device, "analytic"
            )
        jacobian_timing = JacobianTiming(
            seed=seed,
            method=method,
            autodiff_ct_ms=autodiff_ct[0],
            autodiff_ct_std_ms=autodiff_ct[1],
            analytic_ct_ms=analytic_ct[0],
            analytic_ct_std_ms=analytic_ct[1],
            analytic_speedup=autodiff_ct[0] / analytic_ct[0],
        )

    result = Result(
        seed=seed,
        method=method,
        rmse=rmse,
        violation_rate=violation_rate,
        maximum_violation=maximum_violation,
        mean_violation=mean_violation,
        ct_ms=ct_ms,
        ct_std_ms=ct_std_ms,
        mean_iterations=depth.float().mean().item(),
    )
    return (
        result,
        prediction.detach().cpu().numpy().reshape(-1),
        jacobian_timing,
    )


def save_results(results: Iterable[Result], output_dir: Path) -> None:
    rows = [asdict(result) for result in results]
    with (output_dir / "metrics_per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    metric_names = (
        "rmse", "violation_rate", "maximum_violation",
        "mean_violation", "ct_ms", "mean_iterations",
    )
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        summary[method] = {}
        for metric in metric_names:
            values = np.asarray([row[metric] for row in method_rows], dtype=float)
            summary[method][metric] = {
                "mean": float(values.mean()),
                "std_across_seeds": (
                    float(values.std(ddof=1)) if len(values) > 1 else 0.0
                ),
            }
    with (output_dir / "metrics_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    summary_rows = []
    for method in METHODS:
        row = {"method": method}
        for metric in metric_names:
            row[f"{metric}_mean"] = summary[method][metric]["mean"]
            row[f"{metric}_std_across_seeds"] = summary[method][metric][
                "std_across_seeds"
            ]
        summary_rows.append(row)
    with (output_dir / "metrics_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)


def save_jacobian_timings(
    timings: Iterable[JacobianTiming], output_dir: Path
) -> None:
    rows = [asdict(timing) for timing in timings]
    if not rows:
        return
    with (output_dir / "jacobian_timing.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for method in ("ENFORCE", "DiffSlack"):
        method_rows = [row for row in rows if row["method"] == method]
        summary_row = {"method": method}
        for metric in (
            "autodiff_ct_ms", "analytic_ct_ms", "analytic_speedup"
        ):
            values = np.asarray([row[metric] for row in method_rows], dtype=float)
            summary_row[f"{metric}_mean"] = float(values.mean())
            summary_row[f"{metric}_std_across_seeds"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        summary_rows.append(summary_row)
    with (output_dir / "jacobian_timing_summary.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)


def print_jacobian_timings(timings: Iterable[JacobianTiming]) -> None:
    timings = list(timings)
    if not timings:
        return
    print("\n=== Analytic versus autodiff Jacobian CT ===")
    print(
        f"{'Seed':>4}  {'Method':<12}  {'Autodiff (ms)':>22}  "
        f"{'Analytic (ms)':>22}  {'Speedup':>9}"
    )
    for timing in timings:
        print(
            f"{timing.seed:>4d}  {timing.method:<12}  "
            f"{timing.autodiff_ct_ms:>9.4f} ± "
            f"{timing.autodiff_ct_std_ms:<8.4f}  "
            f"{timing.analytic_ct_ms:>9.4f} ± "
            f"{timing.analytic_ct_std_ms:<8.4f}  "
            f"{timing.analytic_speedup:>8.2f}x"
        )

    seeds = sorted({timing.seed for timing in timings})
    if len(seeds) > 1:
        print("\n=== Jacobian CT mean ± standard deviation across seeds ===")
        for method in ("ENFORCE", "DiffSlack"):
            method_timings = [t for t in timings if t.method == method]
            autodiff = np.asarray([t.autodiff_ct_ms for t in method_timings])
            analytic = np.asarray([t.analytic_ct_ms for t in method_timings])
            speedup = np.asarray([t.analytic_speedup for t in method_timings])
            print(
                f"{method:<12}  autodiff={autodiff.mean():.4f} ± "
                f"{autodiff.std(ddof=1):.4f} ms, analytic="
                f"{analytic.mean():.4f} ± {analytic.std(ddof=1):.4f} ms, "
                f"speedup={speedup.mean():.2f} ± "
                f"{speedup.std(ddof=1):.2f}x"
            )


def print_results(results: Iterable[Result]) -> None:
    results = list(results)
    print("\n=== Nonlinear constrained regression results ===")
    print(
        f"{'Seed':>4}  {'Method':<12}  {'RMSE':>10}  {'Viol. rate':>11}  "
        f"{'Max viol.':>10}  {'Mean viol.':>11}  {'CT (ms)':>18}  {'Iter.':>7}"
    )
    for result in results:
        print(
            f"{result.seed:>4d}  {result.method:<12}  {result.rmse:>10.6f}  "
            f"{100.0 * result.violation_rate:>10.2f}%  "
            f"{result.maximum_violation:>10.3e}  "
            f"{result.mean_violation:>11.3e}  "
            f"{result.ct_ms:>8.4f} ± {result.ct_std_ms:<7.4f}  "
            f"{result.mean_iterations:>7.2f}"
        )

    seeds = sorted({result.seed for result in results})
    if len(seeds) > 1:
        print("\n=== Mean ± standard deviation across seeds ===")
        print(
            f"{'Method':<12}  {'RMSE':>21}  {'Viol. rate (%)':>21}  "
            f"{'Max violation':>21}  {'Mean violation':>21}  {'CT (ms)':>21}"
        )
        for method in METHODS:
            method_results = [r for r in results if r.method == method]

            def mean_std(attribute: str, scale: float = 1.0):
                values = np.asarray(
                    [getattr(result, attribute) * scale for result in method_results]
                )
                return values.mean(), values.std(ddof=1)

            rmse = mean_std("rmse")
            rate = mean_std("violation_rate", 100.0)
            maximum = mean_std("maximum_violation")
            mean_violation = mean_std("mean_violation")
            ct = mean_std("ct_ms")
            print(
                f"{method:<12}  {rmse[0]:>9.6f} ± {rmse[1]:<9.6f}  "
                f"{rate[0]:>9.3f} ± {rate[1]:<8.3f}  "
                f"{maximum[0]:>9.3e} ± {maximum[1]:<9.3e}  "
                f"{mean_violation[0]:>9.3e} ± {mean_violation[1]:<9.3e}  "
                f"{ct[0]:>9.4f} ± {ct[1]:<9.4f}"
            )


def save_plot(
    x_test: torch.Tensor,
    predictions: Dict[str, np.ndarray],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    violation_tolerance: float,
    output_dir: Path,
) -> None:
    x = x_test.detach().cpu().numpy().reshape(-1)
    target = nominal_target(x_test).detach().cpu().numpy().reshape(-1)
    exact = exact_constrained_optimum(x_test).detach().cpu().numpy().reshape(-1)
    lower, upper = feasible_bounds(x_test)
    lower = lower.detach().cpu().numpy().reshape(-1)
    upper = upper.detach().cpu().numpy().reshape(-1)
    x_train_np = x_train.detach().cpu().numpy().reshape(-1)
    y_train_np = y_train.detach().cpu().numpy().reshape(-1)

    def residuals(x_value: np.ndarray, y_value: np.ndarray):
        first = np.exp(y_value) - (
            1.85 + 0.32 * np.sin(1.35 * x_value)
        )
        second = (
            0.08 + 0.18 * np.cos(1.15 * x_value)
            - np.log(np.maximum(y_value + 1.45, 1e-12))
        )
        return first, second

    train_g1, train_g2 = residuals(x_train_np, y_train_np)
    train_feasible = np.maximum(train_g1, train_g2) <= violation_tolerance
    method_order = ("DiffSlack", "ENFORCE", "MLP", "Soft penalty")
    colors = {
        "DiffSlack": "#1f77b4",
        "ENFORCE": "#ff7f0e",
        "MLP": "#9467bd",
        "Soft penalty": "#2ca02c",
    }
    residual_by_method = {
        method: residuals(x, predictions[method]) for method in method_order
    }
    top_values = [lower, upper, target, exact, y_train_np]
    top_values.extend(predictions[method] for method in method_order)
    top_min = min(float(values.min()) for values in top_values)
    top_max = max(float(values.max()) for values in top_values)
    top_padding = max(0.05, 0.04 * (top_max - top_min))
    top_limits = (top_min - top_padding, top_max + top_padding)

    style = {
        "font.family": "Arial",
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.1))

        ax = axes[0, 1]
        ax.fill_between(
            x, lower, upper, color="#d9d9d9", alpha=0.55,
            label="Feasible corridor",
        )
        ax.plot(x, upper, "k--", linewidth=1.25, label="Constraint boundaries")
        ax.plot(x, lower, "k--", linewidth=1.25)
        ax.plot(x, exact, color="black", linewidth=2.0,
                label="Exact constrained optimum")
        for method in method_order:
            ax.plot(x, predictions[method], color=colors[method], linewidth=1.7,
                    label=method)
        ax.set_title("Predictions vs feasible corridor")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_ylim(*top_limits)
        ax.legend(loc="upper right", framealpha=0.88)

        ax = axes[0, 0]
        ax.fill_between(x, lower, upper, color="#d9d9d9", alpha=0.55)
        ax.plot(x, upper, "k--", linewidth=1.25)
        ax.plot(x, lower, "k--", linewidth=1.25)
        ax.plot(x, target, color="black", linewidth=1.8, label="Nominal target")
        ax.scatter(
            x_train_np[train_feasible], y_train_np[train_feasible],
            s=9, color="#5a9bd4", alpha=0.45, edgecolors="none",
            label="Feasible labels",
        )
        violation_count = int((~train_feasible).sum())
        ax.scatter(
            x_train_np[~train_feasible], y_train_np[~train_feasible],
            s=15, color="red", alpha=0.82, edgecolors="none",
            label=(
                f"Violating labels "
                f"({100.0 * violation_count / len(x_train_np):.1f}%)"
            ),
        )
        ax.set_title(
            f"Training data: {violation_count}/{len(x_train_np)} labels "
            "violate corridor"
        )
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_ylim(*top_limits)
        ax.legend(loc="upper right", framealpha=0.88)

        residual_titles = (
            r"Upper constraint residual $g_1 \leq 0$",
            r"Lower constraint residual $g_2 \leq 0$",
        )
        for constraint_index, ax in enumerate(axes[1]):
            all_residuals = [
                residual_by_method[method][constraint_index]
                for method in method_order
            ]
            residual_min = min(float(values.min()) for values in all_residuals)
            residual_max = max(float(values.max()) for values in all_residuals)
            padding = max(0.05, 0.04 * (residual_max - residual_min))
            lower_limit = min(-padding, residual_min - padding)
            upper_limit = max(padding, residual_max + padding)
            ax.axhspan(0.0, upper_limit, color="red", alpha=0.07,
                       label=f"Infeasible ($g_{constraint_index + 1}>0$)")
            ax.axhline(
                0.0, color="red", linestyle="--", linewidth=1.3,
                label=rf"$g_{constraint_index + 1}=0$ (boundary)",
            )
            for method in method_order:
                values = residual_by_method[method][constraint_index]
                violation_count = int(
                    np.count_nonzero(values > violation_tolerance)
                )
                ax.plot(
                    x, values, color=colors[method], linewidth=1.6,
                    label=(
                        f"{method} — violations: "
                        f"{violation_count}/{len(x)}"
                    ),
                )
            ax.set_ylim(lower_limit, upper_limit)
            ax.set_title(residual_titles[constraint_index])
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(rf"$g_{constraint_index + 1}(x,\hat{{y}})$")
            legend_location = (
                "lower left" if constraint_index == 0 else "lower right"
            )
            ax.legend(loc=legend_location, framealpha=0.88)

        for ax in axes.flat:
            ax.grid(alpha=0.18, linewidth=0.6)
        fig.tight_layout(h_pad=1.6, w_pad=1.7)
        fig.savefig(output_dir / "predictions.png", dpi=220,
                    bbox_inches="tight")
        fig.savefig(output_dir / "predictions.pdf", bbox_inches="tight")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="demo_results")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--n-train", type=int, default=1200)
    parser.add_argument("--n-test", type=int, default=1500)
    parser.add_argument("--noise-std", type=float, default=0.10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--stage1-epochs", type=int, default=500)
    parser.add_argument("--stage2-epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--penalty-weight", type=float, default=10.0)
    parser.add_argument(
        "--soft-penalty-weight", type=float, default=None,
        help="Override --penalty-weight for Soft penalty only.",
    )
    parser.add_argument(
        "--enforce-penalty-weight", type=float, default=None,
        help="Override --penalty-weight for ENFORCE only.",
    )
    parser.add_argument(
        "--diffslack-penalty-weight", type=float, default=None,
        help="Override --penalty-weight for DiffSlack only.",
    )
    parser.add_argument("--projection-weight", type=float, default=10.0)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--violation-tolerance", type=float, default=1e-6)
    parser.add_argument("--damping", type=float, default=1e-4)
    parser.add_argument("--path-weight", type=float, default=5.0)
    parser.add_argument("--slack-weight", type=float, default=1.0)
    parser.add_argument("--fb-epsilon", type=float, default=1e-12)
    parser.add_argument(
        "--jacobian-mode", choices=("analytic", "autodiff"), default="analytic"
    )
    parser.add_argument(
        "--compare-jacobian-ct",
        dest="compare_jacobian_ct",
        action="store_true",
        default=True,
        help=(
            "Time analytic and autodiff Jacobians on the same trained "
            "ENFORCE and DiffSlack models."
        ),
    )
    parser.add_argument(
        "--no-compare-jacobian-ct",
        dest="compare_jacobian_ct",
        action="store_false",
        help="Skip the paired analytic/autodiff Jacobian timing comparison.",
    )
    parser.add_argument("--ct-samples", type=int, default=500)
    parser.add_argument("--warmup-samples", type=int, default=50)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dtype", choices=("float32", "float64"), default="float64"
    )
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2)

    all_results = []
    all_jacobian_timings = []
    plot_predictions = {}
    plot_training_data = None
    x_test = torch.linspace(
        0.0, 6.0, args.n_test, device=device, dtype=dtype
    ).unsqueeze(-1)
    lower_test, upper_test = feasible_bounds(x_test)
    if torch.any(lower_test > upper_test):
        raise ValueError(
            "The selected constraint parameters produce an empty feasible interval."
        )

    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        x_train, y_train = make_training_data(
            seed, args.n_train, args.noise_std, dtype
        )
        if seed == args.seeds[0]:
            plot_training_data = (x_train.clone(), y_train.clone())
            train_violation = torch.relu(
                inequalities(x_train, y_train)
            ).max(dim=-1).values
            nominal_violation = torch.relu(
                inequalities(x_test.cpu(), nominal_target(x_test.cpu()))
            ).max(dim=-1).values
            print(
                "Problem statistics: "
                f"nominal-target violation rate="
                f"{100.0 * (nominal_violation > args.violation_tolerance).float().mean().item():.2f}%, "
                f"noisy-label violation rate="
                f"{100.0 * (train_violation > args.violation_tolerance).float().mean().item():.2f}%"
            )
        x_train, y_train = x_train.to(device), y_train.to(device)

        for method in METHODS:
            model = train_method(
                method, seed, x_train, y_train, args, device
            )
            result, prediction, jacobian_timing = evaluate_method(
                method, model, x_test, args, seed, device
            )
            all_results.append(result)
            if jacobian_timing is not None:
                all_jacobian_timings.append(jacobian_timing)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "method": method,
                    "seed": seed,
                    "config": vars(args),
                },
                output_dir / f"{method.lower().replace(' ', '_')}_seed_{seed}.pth",
            )
            if seed == args.seeds[0]:
                plot_predictions[method] = prediction

    print_results(all_results)
    print_jacobian_timings(all_jacobian_timings)
    save_results(all_results, output_dir)
    save_jacobian_timings(all_jacobian_timings, output_dir)
    if plot_training_data is None:
        raise RuntimeError("No training data were generated for plotting")
    save_plot(
        x_test,
        plot_predictions,
        plot_training_data[0],
        plot_training_data[1],
        args.violation_tolerance,
        output_dir,
    )
    print(f"\nResults saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
