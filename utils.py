import os

import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path):
    """Create directory if it does not exist."""
    if path and (not os.path.exists(path)):
        os.makedirs(path, exist_ok=True)


def softplus(x):
    """Stable softplus: log(1 + exp(x))."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def sigmoid(x):
    """Stable sigmoid with input clipping."""
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def save_txt(path, arr, header=None):
    """Save array-like data to a text file."""
    ensure_dir(os.path.dirname(path))
    if header is None:
        np.savetxt(path, np.array(arr))
    else:
        np.savetxt(path, np.array(arr), header=header)


def plot_loss(loss_hist, path):
    """Plot loss history vs iteration."""
    ensure_dir(os.path.dirname(path))

    plt.figure(figsize=(7, 4))
    plt.plot(loss_hist, lw=2)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_regression(path, t, y_pred, dy_pred, x, y, dy, title="ProbNN"):
    """Plot predictive mean with ±1σ band and training error bars."""
    ensure_dir(os.path.dirname(path))

    t = np.array(t).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    dy_pred = np.array(dy_pred).reshape(-1)

    x = np.array(x).reshape(-1)
    y = np.array(y).reshape(-1)
    dy = np.array(dy).reshape(-1)

    plt.figure(figsize=(8, 4.5))
    plt.fill_between(
        t,
        y_pred - dy_pred,
        y_pred + dy_pred,
        alpha=0.25,
        label="±1σ (model)",
    )
    plt.plot(t, y_pred, lw=2, label="prediction")
    plt.errorbar(x, y, yerr=dy, fmt=".", capsize=3, label="train data")

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def normalized_residuals(x, y, dy, mu_train, sigma_train):
    """Compute r = (y - mu) / sqrt(dy^2 + sigma^2) on training inputs."""
    x = np.array(x).reshape(-1)
    y = np.array(y).reshape(-1)
    dy = np.array(dy).reshape(-1)
    mu_train = np.array(mu_train).reshape(-1)
    sigma_train = np.array(sigma_train).reshape(-1)

    v = dy**2 + sigma_train**2
    r = (y - mu_train) / np.sqrt(v)
    return r


def plot_residuals(path, r, title="Normalized residuals"):
    """Plot histogram of normalized residuals."""
    ensure_dir(os.path.dirname(path))

    r = np.array(r).reshape(-1)

    plt.figure(figsize=(7, 4))
    plt.hist(r, bins=18, alpha=0.8)
    plt.title(title)
    plt.xlabel("r = (y - mu) / sqrt(dy^2 + sigma^2)")
    plt.ylabel("count")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
