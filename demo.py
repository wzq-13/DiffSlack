import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# ============================================================
# Nonlinear constrained regression
#
# Input:
#   x in [0, 6]
#
# Output:
#   y
#
# Objective:
#   stay close to the nominal target f(x)
#
# Constraints:
#   g1(x,y) = exp(y) - (1.85 + 0.32 sin(1.35x)) <= 0
#
#   g2(x,y) = 0.08 + 0.18 cos(1.15x)
#             - log(y + 1.45) <= 0
#
# Equivalent feasible interval:
#
#   exp(0.08 + 0.18 cos(1.15x)) - 1.45
#        <= y <=
#   log(1.85 + 0.32 sin(1.35x))
#
# Training noise is added ONLY to y.
# ============================================================


# ------------------------------------------------------------
# 1. Nominal target
# ------------------------------------------------------------

def target(x):
    return (
        0.65 * np.sin(x)
        + 0.20 * np.log(1.0 + x)
    )


# ------------------------------------------------------------
# 2. Nonlinear constraints
# ------------------------------------------------------------

def g1(x, y):
    """
    Upper nonlinear inequality:
        exp(y) - (1.85 + 0.32 sin(1.35x)) <= 0
    """
    return np.exp(y) - (1.85 + 0.32 * np.sin(1.35 * x))


def g2(x, y):
    """
    Lower nonlinear inequality:
        0.08 + 0.18 cos(1.15x) - log(y + 1.45) <= 0
    """
    return 0.08 + 0.18 * np.cos(1.15 * x) - np.log(y + 1.45)


def upper_bound(x):
    return np.log(1.85 + 0.32 * np.sin(1.35 * x))


def lower_bound(x):
    return np.exp(0.08 + 0.18 * np.cos(1.15 * x)) - 1.45


def is_feasible(x, y):
    valid_domain = y > -1.45

    feasible = np.zeros_like(y, dtype=bool)

    feasible[valid_domain] = (
        (g1(x[valid_domain], y[valid_domain]) <= 0.0)
        &
        (g2(x[valid_domain], y[valid_domain]) <= 0.0)
    )

    return feasible


# ============================================================
# 3. Dense curves
# ============================================================

x_plot = np.linspace(0.0, 6.0, 1500)

y_target = target(x_plot)

y_upper = upper_bound(x_plot)
y_lower = lower_bound(x_plot)

# Exact pointwise constrained optimum
y_exact = np.clip(
    y_target,
    y_lower,
    y_upper
)

target_feasible = (
    (y_target >= y_lower)
    &
    (y_target <= y_upper)
)

target_violation_ratio = (
    1.0 - np.mean(target_feasible)
)

upper_violation_ratio = np.mean(
    y_target > y_upper
)

lower_violation_ratio = np.mean(
    y_target < y_lower
)

print(
    f"Nominal target violation ratio: "
    f"{100 * target_violation_ratio:.2f}%"
)

print(
    f"Upper violation: "
    f"{100 * upper_violation_ratio:.2f}%"
)

print(
    f"Lower violation: "
    f"{100 * lower_violation_ratio:.2f}%"
)


# ============================================================
# 4. Training data
# ============================================================

N_train = 1200

# Input is sampled but NOT perturbed
x_train = np.random.uniform(
    0.0,
    6.0,
    N_train
)

y_train_clean = target(x_train)

# Noise ONLY on the output
sigma_y = 0.10

y_train_noisy = (
    y_train_clean
    + np.random.normal(
        0.0,
        sigma_y,
        N_train
    )
)

train_feasible = is_feasible(
    x_train,
    y_train_noisy
)

train_violation_ratio = (
    1.0 - np.mean(train_feasible)
)

print(
    f"Noisy training-label violation ratio: "
    f"{100 * train_violation_ratio:.2f}%"
)


# ============================================================
# 5. Plot
# ============================================================

plot_style = {
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 10,
}

with plt.rc_context(plot_style):
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.1))
    top_min = min(y_lower.min(), y_target.min(), y_exact.min(), y_train_noisy.min())
    top_max = max(y_upper.max(), y_target.max(), y_exact.max(), y_train_noisy.max())
    top_padding = max(0.05, 0.04 * (top_max - top_min))
    top_limits = (top_min - top_padding, top_max + top_padding)

    ax = axes[0, 0]
    ax.fill_between(x_plot, y_lower, y_upper, color="#d9d9d9", alpha=0.55,
                    label="Feasible corridor")
    ax.plot(x_plot, y_upper, "k--", linewidth=1.25,
            label="Constraint boundaries")
    ax.plot(x_plot, y_lower, "k--", linewidth=1.25)
    ax.plot(x_plot, y_exact, color="black", linewidth=2.0,
            label="Exact constrained optimum")
    ax.set_title("Target vs feasible corridor")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_ylim(*top_limits)
    ax.legend(loc="upper right", framealpha=0.88)

    ax = axes[0, 1]
    ax.fill_between(x_plot, y_lower, y_upper, color="#d9d9d9", alpha=0.55)
    ax.plot(x_plot, y_upper, "k--", linewidth=1.25)
    ax.plot(x_plot, y_lower, "k--", linewidth=1.25)
    ax.plot(x_plot, y_target, color="black", linewidth=1.8,
            label="Nominal target")
    ax.scatter(x_train[train_feasible], y_train_noisy[train_feasible], s=9,
               color="#5a9bd4", alpha=0.45, edgecolors="none",
               label="Feasible labels")
    violation_count = int((~train_feasible).sum())
    ax.scatter(x_train[~train_feasible], y_train_noisy[~train_feasible], s=15,
               color="red", alpha=0.82, edgecolors="none",
               label=f"Violating labels ({100*train_violation_ratio:.1f}%)")
    ax.set_title(
        f"Training data: {violation_count}/{N_train} labels violate corridor"
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_ylim(*top_limits)
    ax.legend(loc="upper right", framealpha=0.88)

    curve_definitions = (
        (g1(x_plot, y_target), g1(x_plot, y_exact),
         r"Upper constraint residual $g_1 \leq 0$", r"$g_1(x,y)$"),
        (g2(x_plot, y_target), g2(x_plot, y_exact),
         r"Lower constraint residual $g_2 \leq 0$", r"$g_2(x,y)$"),
    )
    for constraint_index, (ax, curve_definition) in enumerate(zip(
        axes[1], curve_definitions
    )):
        nominal_residual, exact_residual, title, ylabel = curve_definition
        residual_min = min(nominal_residual.min(), exact_residual.min())
        residual_max = max(nominal_residual.max(), exact_residual.max())
        padding = max(0.05, 0.04 * (residual_max - residual_min))
        lower_limit = min(-padding, residual_min - padding)
        upper_limit = max(padding, residual_max + padding)
        ax.axhspan(0.0, upper_limit, color="red", alpha=0.07,
                   label="Infeasible ($g>0$)")
        ax.axhline(0.0, color="red", linestyle="--", linewidth=1.3,
                   label="$g=0$ (boundary)")
        ax.plot(x_plot, nominal_residual, color="#9467bd", linewidth=1.7,
                label="Nominal target")
        ax.plot(x_plot, exact_residual, color="#1f77b4", linewidth=1.8,
                label="Exact constrained optimum")
        ax.set_ylim(lower_limit, upper_limit)
        ax.set_title(title)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(ylabel)
        legend_location = (
            "lower left" if constraint_index == 0 else "lower right"
        )
        ax.legend(loc=legend_location, framealpha=0.88)

    for ax in axes.flat:
        ax.grid(alpha=0.18, linewidth=0.6)
    fig.tight_layout(h_pad=1.6, w_pad=1.7)
    fig.savefig("demo_problem.png", dpi=220, bbox_inches="tight")
    fig.savefig("demo_problem.pdf", bbox_inches="tight")
    plt.show()
