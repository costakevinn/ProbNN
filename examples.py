import numpy as np
import matplotlib.pyplot as plt

from activation import Tanh
from nn import regression, predict_train
from utils import (
    save_txt,
    plot_loss,
    plot_regression,
    normalized_residuals,
    plot_residuals,
)


def gaussian_noise(n, std=1.0):
    """Generate Gaussian noise via Box–Muller (size n, std)."""
    m = (n + 1) // 2
    u1 = np.random.rand(m)
    u2 = np.random.rand(m)
    z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    z1 = np.sqrt(-2.0 * np.log(u1)) * np.sin(2.0 * np.pi * u2)
    return np.concatenate([z0, z1])[:n] * std


def real_function(x):
    """Baseline ground-truth used only in synthetic benchmarks."""
    return np.sin(x) + 0.5 * x


def plot_benchmark(filename, t, y_pred, dy_pred, x, y, dy, y_true, title):
    """Benchmark-only plot with truth overlay (not part of the tool interface)."""
    plt.figure(figsize=(8, 4.5))

    plt.plot(t, y_true, "--", lw=2, label="True function (benchmark)")
    plt.plot(t, y_pred, lw=2, label="ProbNN mean")
    plt.fill_between(
        t,
        y_pred - dy_pred,
        y_pred + dy_pred,
        alpha=0.30,
        label="ProbNN uncertainty",
    )

    plt.errorbar(x, y, yerr=dy, fmt=".", capsize=3, label="Observations")

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def example_sine(seed=450):
    """Example 1: smooth baseline regression (sin + linear trend)."""
    np.random.seed(seed)

    n = 10
    x = np.linspace(0.0, 5.0, n) + gaussian_noise(n, std=1.0)
    y_true = real_function(x)

    y = y_true + gaussian_noise(n, std=0.20)
    dy = np.abs(gaussian_noise(n, std=0.25)) + 1e-2

    t_predict = np.linspace(np.min(x) - 5.0, np.max(x) + 5.0, 600)

    activation = Tanh()

    y_pred, dy_pred, loss_hist, params = regression(
        t_predict,
        x, y, dy,
        layers=[32, 32, 32, 32],
        activation=activation,
        lr=1e-3,
        n_iters=6000,
        seed=0,
        lam_sigma=1e-3,
        lam_l2=0.0,
        return_history=True,
        return_params=True,
    )

    save_txt(
        "results/pred_sine.txt",
        np.column_stack([t_predict, y_pred, dy_pred]),
        header="t  y_pred  dy_pred",
    )
    save_txt("results/loss_sine.txt", np.array(loss_hist), header="loss")

    plot_loss(loss_hist, "plots/loss_sine.png")

    # Tool-style output (real-world usage)
    plot_regression(
        "plots/fit_sine.png",
        t_predict, y_pred, dy_pred,
        x, y, dy,
        title="ProbNN: sin(x) + 0.5x",
    )

    # Benchmark-only diagnostic (truth overlay)
    y_true_predict = real_function(t_predict)
    plot_benchmark(
        "plots/fit_sine_truth.png",
        t_predict, y_pred, dy_pred,
        x, y, dy,
        y_true_predict,
        title="ProbNN (benchmark): sin(x) + 0.5x",
    )

    mu_train, sigma_train = predict_train(x, params, activation)
    r = normalized_residuals(x, y, dy, mu_train, sigma_train)

    save_txt("results/residuals_sine.txt", r, header="normalized_residuals")
    plot_residuals("plots/residuals_sine.png", r, title="Residuals: sine")

    print("Saved: plots/*, results/* (sine)")


def example_generic(seed=999):
    """Example 2: generic smooth multi-scale function (edit f(x))."""
    np.random.seed(seed)

    def f(x):
        return np.sin(x) + 0.5 * np.sin(3.0 * x) + 0.2 * x

    n = 20
    x = np.linspace(-2.0, 4.0, n) + gaussian_noise(n, std=0.10)
    y_true = f(x)

    dy = 0.15 + 0.05 * np.abs(gaussian_noise(n, std=1.0))
    y = y_true + gaussian_noise(n, std=1.0) * dy

    t_predict = np.linspace(np.min(x) - 4.0, np.max(x) + 4.0, 900)

    activation = Tanh()

    y_pred, dy_pred, loss_hist, params = regression(
        t_predict,
        x, y, dy,
        layers=[64, 64, 64],
        activation=activation,
        lr=1e-3,
        n_iters=8000,
        seed=0,
        lam_sigma=1e-3,
        lam_l2=0.0,
        return_history=True,
        return_params=True,
    )

    save_txt(
        "results/pred_generic.txt",
        np.column_stack([t_predict, y_pred, dy_pred]),
        header="t  y_pred  dy_pred",
    )
    save_txt("results/loss_generic.txt", np.array(loss_hist), header="loss")

    plot_loss(loss_hist, "plots/loss_generic.png")

    plot_regression(
        "plots/fit_generic.png",
        t_predict, y_pred, dy_pred,
        x, y, dy,
        title="ProbNN: generic function (edit f(x))",
    )

    y_true_predict = f(t_predict)
    plot_benchmark(
        "plots/fit_generic_truth.png",
        t_predict, y_pred, dy_pred,
        x, y, dy,
        y_true_predict,
        title="ProbNN (benchmark): generic function",
    )

    mu_train, sigma_train = predict_train(x, params, activation)
    r = normalized_residuals(x, y, dy, mu_train, sigma_train)

    save_txt("results/residuals_generic.txt", r, header="normalized_residuals")
    plot_residuals("plots/residuals_generic.png", r, title="Residuals: generic")

    print("Saved: plots/*, results/* (generic)")


def example_discontinuous(seed=321):
    """Example 3: discontinuous target (jump + smooth component)."""
    np.random.seed(seed)

    def f(x):
        x0 = 0.0
        A = 1.1
        return np.sin(3.0 * x) + 0.3 * x + A * (x >= x0)

    n = 40
    x = np.linspace(-4.0, 4.0, n) + gaussian_noise(n, std=0.08)
    y_true = f(x)

    dy = 0.10 + 0.04 * np.abs(gaussian_noise(n, std=1.0))
    y = y_true + gaussian_noise(n, std=1.0) * dy

    t_predict = np.linspace(np.min(x) - 2.0, np.max(x) + 2.0, 900)

    activation = Tanh()

    # Stability knobs for discontinuities
    y_pred, dy_pred, loss_hist, params = regression(
        t_predict,
        x, y, dy,
        layers=[64, 64, 64],
        activation=activation,
        lr=3e-4,
        n_iters=12000,
        seed=0,
        lam_sigma=2e-2,
        lam_l2=1e-6,
        return_history=True,
        return_params=True,
    )

    save_txt(
        "results/pred_discontinuous.txt",
        np.column_stack([t_predict, y_pred, dy_pred]),
        header="t  y_pred  dy_pred",
    )
    save_txt("results/loss_discontinuous.txt", np.array(loss_hist), header="loss")

    plot_loss(loss_hist, "plots/loss_discontinuous.png")

    plot_regression(
        "plots/fit_discontinuous.png",
        t_predict, y_pred, dy_pred,
        x, y, dy,
        title="ProbNN: discontinuous target (jump + smooth)",
    )

    y_true_predict = f(t_predict)
    plot_benchmark(
        "plots/fit_discontinuous_truth.png",
        t_predict, y_pred, dy_pred,
        x, y, dy,
        y_true_predict,
        title="ProbNN (benchmark): discontinuous target",
    )

    mu_train, sigma_train = predict_train(x, params, activation)
    r = normalized_residuals(x, y, dy, mu_train, sigma_train)

    save_txt("results/residuals_discontinuous.txt", r, header="normalized_residuals")
    plot_residuals("plots/residuals_discontinuous.png", r, title="Residuals: discontinuous")

    print("Saved: plots/*, results/* (discontinuous)")


def run_all():
    """Run all benchmark examples."""
    example_sine()
    example_generic()
    example_discontinuous()
