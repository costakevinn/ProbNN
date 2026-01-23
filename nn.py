import numpy as np

from utils import softplus, sigmoid


def gaussian_nll(mu, s, y, dy, lam_sigma=0.0, eps=1e-12):
    """Gaussian NLL with heteroscedastic observational noise and learned model noise."""
    n = len(y)

    sigma = softplus(s) + 1e-6
    v = dy**2 + sigma**2 + eps
    diff = y - mu

    nll = 0.5 * np.mean(diff**2 / v + np.log(v))

    loss = nll
    if lam_sigma > 0.0:
        loss = loss + lam_sigma * np.mean(s**2)

    dmu = (-diff / v) / n

    dL_dsigma = (1.0 / v - diff**2 / v**2) * sigma
    ds = (dL_dsigma * sigmoid(s)) / n

    if lam_sigma > 0.0:
        ds = ds + (2.0 * lam_sigma * s) / n

    return float(loss), dmu, ds


def train(
    x_train,
    y_train,
    dy_train,
    layers,
    activation,
    lr=1e-3,
    n_iters=5000,
    seed=0,
    lam_sigma=1e-3,
    lam_l2=0.0,
    eps=1e-12,
):
    """Train a dense trunk with two heads: mean (mu) and log-std proxy (s)."""
    np.random.seed(seed)

    x = np.array(x_train).reshape(-1, 1)
    y = np.array(y_train).reshape(-1, 1)
    dy = np.array(dy_train).reshape(-1, 1)

    if len(x) != len(y) or len(x) != len(dy):
        raise ValueError("x_train, y_train, dy_train must have same length")

    sizes = [1] + list(layers)
    L = len(sizes) - 1  # trunk layer count

    # Trunk parameters
    W, b = [], []
    for i in range(L):
        fan_in = sizes[i]
        scale = np.sqrt(2.0 / fan_in)
        W.append(scale * np.random.randn(fan_in, sizes[i + 1]))
        b.append(np.zeros((1, sizes[i + 1])))

    # Head parameters (mu, s)
    H = sizes[-1]
    np.random.seed(seed + 123)

    W_mu = np.sqrt(2.0 / H) * np.random.randn(H, 1)
    b_mu = np.zeros((1, 1))

    W_s = np.sqrt(2.0 / H) * np.random.randn(H, 1)
    b_s = np.zeros((1, 1))

    loss_hist = []

    for _ in range(n_iters):
        acts = [activation.__class__() for _ in range(max(0, L - 1))]

        # Forward (trunk)
        A = [x]
        a = x
        for i in range(L):
            z = a @ W[i] + b[i]
            if i == L - 1:
                a = z
            else:
                a = acts[i].forward(z)
            A.append(a)

        h = a

        # Forward (heads)
        mu = h @ W_mu + b_mu
        s = h @ W_s + b_s

        loss, dmu, ds = gaussian_nll(mu, s, y, dy, lam_sigma=lam_sigma, eps=eps)
        loss_hist.append(loss)

        if not np.isfinite(loss):
            raise ValueError("Loss became NaN/Inf. Reduce lr or increase lam_sigma.")

        # Head gradients
        dW_mu = h.T @ dmu
        db_mu = np.sum(dmu, axis=0, keepdims=True)
        dh_mu = dmu @ W_mu.T

        dW_s = h.T @ ds
        db_s = np.sum(ds, axis=0, keepdims=True)
        dh_s = ds @ W_s.T

        dh = dh_mu + dh_s

        # Backprop (trunk)
        dW = [None] * L
        db = [None] * L

        dA = dh
        for i in range(L - 1, -1, -1):
            if i == L - 1:
                dZ = dA
            else:
                dZ = acts[i].backward(dA)

            dW[i] = A[i].T @ dZ
            db[i] = np.sum(dZ, axis=0, keepdims=True)
            dA = dZ @ W[i].T

        # L2 regularization
        if lam_l2 > 0.0:
            for i in range(L):
                dW[i] += 2.0 * lam_l2 * W[i]
            dW_mu += 2.0 * lam_l2 * W_mu
            dW_s += 2.0 * lam_l2 * W_s

        # SGD update
        for i in range(L):
            W[i] -= lr * dW[i]
            b[i] -= lr * db[i]

        W_mu -= lr * dW_mu
        b_mu -= lr * db_mu

        W_s -= lr * dW_s
        b_s -= lr * db_s

    params = {
        "W": W,
        "b": b,
        "W_mu": W_mu,
        "b_mu": b_mu,
        "W_s": W_s,
        "b_s": b_s,
        "layers": list(layers),
        "activation": activation.__class__.__name__,
    }
    return params, loss_hist


def predict(t_predict, params, activation):
    """Predict mean and model std at arbitrary inputs t_predict."""
    t = np.array(t_predict).reshape(-1, 1)

    W = params["W"]
    b = params["b"]
    W_mu = params["W_mu"]
    b_mu = params["b_mu"]
    W_s = params["W_s"]
    b_s = params["b_s"]

    L = len(W)
    acts = [activation.__class__() for _ in range(max(0, L - 1))]

    a = t
    for i in range(L):
        z = a @ W[i] + b[i]
        if i == L - 1:
            a = z
        else:
            a = acts[i].forward(z)

    h = a
    mu = h @ W_mu + b_mu
    s = h @ W_s + b_s

    y_pred = mu.reshape(-1)
    dy_pred = (softplus(s) + 1e-6).reshape(-1)

    return y_pred, dy_pred


def predict_train(x_train, params, activation):
    """Predict mean and model std at training inputs (for diagnostics)."""
    x = np.array(x_train).reshape(-1, 1)

    W = params["W"]
    b = params["b"]
    W_mu = params["W_mu"]
    b_mu = params["b_mu"]
    W_s = params["W_s"]
    b_s = params["b_s"]

    L = len(W)
    acts = [activation.__class__() for _ in range(max(0, L - 1))]

    a = x
    for i in range(L):
        z = a @ W[i] + b[i]
        if i == L - 1:
            a = z
        else:
            a = acts[i].forward(z)

    h = a
    mu = h @ W_mu + b_mu
    s = h @ W_s + b_s

    mu_train = mu.reshape(-1)
    sigma_train = (softplus(s) + 1e-6).reshape(-1)

    return mu_train, sigma_train


def regression(
    t_predict,
    x_train,
    y_train,
    dy_train,
    layers,
    activation,
    lr=1e-3,
    n_iters=5000,
    seed=0,
    lam_sigma=1e-3,
    lam_l2=0.0,
    return_history=False,
    return_params=False,
):
    """Fit on (x,y,dy) and return (y_pred, dy_pred) on t_predict."""
    params, loss_hist = train(
        x_train,
        y_train,
        dy_train,
        layers=layers,
        activation=activation,
        lr=lr,
        n_iters=n_iters,
        seed=seed,
        lam_sigma=lam_sigma,
        lam_l2=lam_l2,
    )

    y_pred, dy_pred = predict(t_predict, params, activation)

    out = (y_pred, dy_pred)
    if return_history:
        out = out + (loss_hist,)
    if return_params:
        out = out + (params,)

    return out
