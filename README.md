# 🚀 ProbNN — Probabilistic Neural Network for Heteroscedastic Regression

ProbNN is a probabilistic neural network framework designed for regression with explicit uncertainty quantification.

It models both predictive mean and heteroscedastic uncertainty within a fully differentiable architecture, combining statistical modeling, likelihood-based training, and gradient-driven optimization.

Author: Kevin Mota da Costa
Portfolio: [https://costakevinn.github.io](https://costakevinn.github.io)
LinkedIn: [https://linkedin.com/in/SEUUSER](https://linkedin.com/in/SEUUSER)

---

## 🎯 Project Purpose

ProbNN was developed to explore regression under realistic noise conditions, where:

* Variance changes across the input space
* Discontinuities or sharp transitions exist
* Local and global structures coexist
* Predictive confidence matters as much as accuracy

The project emphasizes probabilistic reasoning, analytical gradients, and controlled optimization dynamics — reflecting a statistical-first approach to machine learning.

---

## 🧠 Probabilistic Model

Given observations:

[
\mathcal{D} = {(x_i, y_i, \delta y_i)}_{i=1}^N
]

ProbNN models the conditional distribution as:

[
p(y \mid x) = \mathcal{N}\big(\mu(x), \delta y^2 + \sigma^2(x)\big)
]

Where:

* (\mu(x)) → predictive mean (neural output)
* (\sigma(x)) → learned model uncertainty
* (\delta y) → known observational noise

This formulation enables heteroscedastic regression, allowing the model to adapt uncertainty locally instead of assuming constant variance.

---

## 🏗 Network Architecture

The architecture consists of:

* Shared dense trunk (feature representation)
* Mean head → predicts (\mu(x))
* Uncertainty head → predicts latent (s(x))

Uncertainty is enforced positive via:

[
\sigma(x) = \text{softplus}(s(x)) + \varepsilon
]

Design considerations:

* Softplus ensures numerical stability
* Regularization prevents variance inflation
* Nonlinear activations (tanh / ReLU) allow multi-scale representation

The architecture is fully differentiable and optimized end-to-end.

---

## 📉 Training Objective

Training minimizes the Gaussian Negative Log-Likelihood (NLL):

[
\mathcal{L}
= \frac{1}{2N} \sum_{i=1}^N
\left[
\frac{(y_i - \mu_i)^2}{\delta y_i^2 + \sigma_i^2}

* \log(\delta y_i^2 + \sigma_i^2)
  \right]
* \lambda_\sigma |s|^2
  ]

This objective balances:

* Data fidelity (residual term)
* Uncertainty calibration (log-variance term)
* Regularization for stability

Optimization is performed via stochastic gradient descent with full backpropagation through:

* Likelihood function
* Softplus transformation
* Activation derivatives
* All network parameters

---

## 🔄 Training Mechanics (System View)

1. Forward pass through trunk
2. Dual-head outputs ((\mu, s))
3. Likelihood evaluation
4. Analytical gradient computation
5. Parameter updates via SGD

This tight coupling between probability theory and optimization is the core engineering design of ProbNN.

---

## 📊 Diagnostics & Evaluation

Model calibration is evaluated using normalized residuals:

[
r_i = \frac{y_i - \mu(x_i)}{\sqrt{\delta y_i^2 + \sigma^2(x_i)}}
]

If properly calibrated, residuals approximate a standard normal distribution.

This provides a principled statistical diagnostic beyond traditional regression metrics.

---

## 🧪 Discontinuous Regression Benchmark

### Predictive Fit

![Discontinuous benchmark](plots/fit_discontinuous_truth.png)

The model successfully captures:

* Global smooth structure
* Sharp discontinuities
* Multi-scale nonlinear behavior
* Increased uncertainty near difficult regions

---

### Training Dynamics

![Training loss](plots/loss_discontinuous.png)

Loss shows stable convergence under likelihood-based optimization, even with discontinuities.

Careful tuning of learning rate and uncertainty regularization ensures numerical robustness.

---

### Residual Calibration

![Residuals](plots/residuals_discontinuous.png)

Residuals remain centered and symmetric, indicating consistent mean estimation and well-calibrated uncertainty.

---

## 📚 Engineering Decisions

* Likelihood-based objective instead of MSE
* Explicit heteroscedastic modeling
* Softplus transformation for stability
* Regularization of variance head
* Analytical gradient propagation
* Modular separation of architecture and diagnostics

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

Generates:

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
