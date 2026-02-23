# 🚀 ProbNN — Probabilistic Neural Network for Uncertainty-Aware Regression

ProbNN is a probabilistic neural network framework for regression with explicit uncertainty modeling.

It jointly learns:

* Predictive mean
* Heteroscedastic (input-dependent) uncertainty

The framework is designed for regression tasks where predictive confidence matters as much as accuracy.

Author: Kevin Mota da Costa
Portfolio: [https://costakevinn.github.io](https://costakevinn.github.io)
LinkedIn: [https://linkedin.com/in/SEUUSER](https://linkedin.com/in/SEUUSER)

---

## 🎯 Project Purpose

ProbNN was developed to explore regression under realistic noise conditions, including:

* Nonlinear functions
* Discontinuities and sharp transitions
* Input-dependent variance
* Multi-scale structure

Instead of minimizing Mean Squared Error, the model is trained via likelihood maximization, enabling principled uncertainty calibration.

This project reflects a statistical-first approach to machine learning systems.

---

## 🧠 Probabilistic Formulation

Given observations (x, y, δy), the model assumes:

p(y | x) = Normal( μ(x), δy² + σ(x)² )

Where:

* μ(x) → predictive mean (neural output)
* σ(x) → learned model uncertainty
* δy → known observational noise

This enables heteroscedastic regression, allowing the model to adapt uncertainty locally rather than assuming constant noise across the dataset.

---

## 🏗 Network Architecture

ProbNN uses:

* Shared dense trunk (feature extractor)
* Mean head → predicts μ(x)
* Uncertainty head → predicts latent s(x)

Uncertainty is mapped using:

σ(x) = softplus(s(x)) + ε

Design choices:

* Softplus ensures positivity and numerical stability
* Separate heads prevent interference between mean and variance learning
* Nonlinear activations (tanh / ReLU) allow multi-scale representation

The entire system is fully differentiable and trained end-to-end.

---

## 📉 Training Objective

The model minimizes the Gaussian Negative Log-Likelihood (NLL):

L = 1/(2N) Σ [ (y − μ)² / (δy² + σ²) + log(δy² + σ²) ] + λ ||s||²

This objective balances:

* Data fidelity (residual term)
* Uncertainty calibration (log-variance term)
* Regularization of the uncertainty head

Optimization is performed via stochastic gradient descent with full backpropagation through:

* Likelihood computation
* Softplus transformation
* Activation derivatives
* All network parameters

---

## 🔄 Training Mechanics (System View)

1. Forward pass through trunk network
2. Dual-head output (mean and variance)
3. Likelihood-based loss evaluation
4. Gradient computation
5. Parameter updates

This tight integration of probability theory and gradient-based optimization is the core design of ProbNN.

---

## 📊 Diagnostics & Evaluation

Model quality is evaluated using normalized residuals:

r = (y − μ(x)) / sqrt(δy² + σ(x)²)

If the model is well calibrated:

* Residuals are centered around zero
* Variance approximates one
* Distribution resembles standard normal

This provides a principled statistical diagnostic beyond simple regression metrics.

---

## 🧪 Discontinuous Regression Benchmark

### Predictive Fit

![Discontinuous benchmark](plots/fit_discontinuous_truth.png)

The model captures:

* Global structure across the full domain
* Local nonlinear behavior
* Sharp discontinuities without oscillatory artifacts
* Increased uncertainty near difficult regions

---

### Training Dynamics

![Training loss](plots/loss_discontinuous.png)

The loss shows stable convergence under a likelihood-based objective, even in the presence of discontinuities.

---

### Residual Calibration

![Residuals](plots/residuals_discontinuous.png)

Residuals remain approximately centered and symmetric, indicating consistent mean estimation and well-calibrated uncertainty.

---

## 📚 Engineering Decisions

* Likelihood-based training instead of MSE
* Explicit heteroscedastic modeling
* Softplus variance mapping for stability
* Regularization to prevent variance collapse
* Modular separation of model and diagnostics

---

## 🛠 Tech Stack

Python
NumPy
Gradient-based optimization
Statistical modeling
Likelihood maximization
Diagnostic visualization

---

## ▶ Usage

```bash
python main.py
```

Runs benchmark examples and generates:

* Predictive fits
* Loss curves
* Residual diagnostics

Outputs are saved to `plots/` and `results/`.

---

## 🌐 Portfolio

This project is part of my Machine Learning portfolio:
👉 [https://costakevinn.github.io](https://costakevinn.github.io)

---

## License

MIT License — see `LICENSE` for details.
