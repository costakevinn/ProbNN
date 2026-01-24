# ProbNN — Probabilistic Neural Network Regression

ProbNN is a **probabilistic neural network framework** for **regression with uncertainty quantification**, designed to model both **predictive mean** and **heteroscedastic uncertainty** using a fully differentiable neural architecture.

The project focuses on **statistical learning**, **gradient-based optimization**, and **probabilistic modeling**, combining classical neural networks with an explicit likelihood-based training objective.  
It is particularly suited for regression problems where **noise varies across the input space**, **local and global structures coexist**, or **discontinuities and sharp transitions** are present.

---

## Probabilistic Regression Model

Given observations  
\[
\mathcal{D} = \{(x_i, y_i, \delta y_i)\}_{i=1}^N,
\]
ProbNN models the conditional distribution of targets as

\[
p(y \mid x) = \mathcal{N}\big(\mu(x), \; \delta y^2 + \sigma^2(x)\big),
\]

where:

- \(\mu(x)\) is the **predictive mean**, learned by a neural network  
- \(\sigma(x)\) is the **model uncertainty**, also learned by the network  
- \(\delta y\) represents known **observational noise**  

This formulation enables **heteroscedastic regression**, allowing the model to adapt uncertainty locally rather than assuming constant noise.

---

## Neural Network Architecture

ProbNN uses a **shared dense trunk** followed by two independent output heads:

- **Mean head**: predicts \(\mu(x)\)  
- **Uncertainty head**: predicts a latent variable \(s(x)\), mapped to \(\sigma(x)\)  

\[
\sigma(x) = \text{softplus}(s(x)) + \varepsilon
\]

The softplus transformation ensures **positivity and numerical stability**, which is essential for probabilistic optimization.

Hidden layers use nonlinear activation functions (e.g. **tanh**, **ReLU**), enabling the network to represent **highly nonlinear functions**, multi-scale behavior, and sharp transitions.

---

## Training Objective and Optimization

Training is performed by minimizing the **Gaussian negative log-likelihood (NLL)**:

\[
\mathcal{L}
= \frac{1}{2N} \sum_{i=1}^N
\left[
\frac{(y_i - \mu_i)^2}{\delta y_i^2 + \sigma_i^2}
+ \log(\delta y_i^2 + \sigma_i^2)
\right]
+ \lambda_\sigma \, \|s\|^2
\]

This objective balances:

- **Data fidelity** via the squared residual term  
- **Uncertainty calibration** via the log-variance term  
- **Regularization** of the uncertainty head to prevent pathological solutions  

Gradients are computed analytically via **backpropagation**, using the chain rule through:

- the likelihood function  
- the softplus and sigmoid nonlinearities  
- the activation functions in each hidden layer  

Parameters (weights and biases) are updated using **stochastic gradient descent**, making the entire training process fully differentiable and scalable.

---

## Forward and Backward Propagation (Conceptual View)

During training:

1. Inputs propagate forward through the trunk network  
2. The network outputs \(\mu(x)\) and \(s(x)\)  
3. The probabilistic loss evaluates data fit and uncertainty  
4. Gradients flow backward:
   - from the likelihood to \(\mu\) and \(\sigma\)
   - through softplus and activation derivatives
   - back into all network weights  

This tight coupling between **optimization**, **probability theory**, and **neural computation** is the core design of ProbNN.

---

## Diagnostic Residuals

Model quality is evaluated using **normalized residuals**:

\[
r_i = \frac{y_i - \mu(x_i)}{\sqrt{\delta y_i^2 + \sigma^2(x_i)}}
\]

If the probabilistic model is well calibrated, residuals should approximately follow a **standard normal distribution**, providing a principled statistical diagnostic beyond visual inspection.

---

## Example: Discontinuous Regression Benchmark

The following example demonstrates ProbNN on a **discontinuous target function** with both smooth and abrupt components — a setting that is challenging for classical regression models.

### Predictive Fit with Ground Truth (Benchmark)

![Discontinuous benchmark](plots/fit_discontinuous_truth.png)

In this example, the network successfully captures:

- **Global structure** of the function across the full domain  
- **Local behavior** near the discontinuity  
- A sharp jump without oscillatory artifacts  
- Increased predictive uncertainty around difficult regions  

This illustrates the model’s ability to reconcile **local non-smooth features** with **global function learning**, a common challenge in real-world regression tasks.

---

### Training Dynamics

![Training loss](plots/loss_discontinuous.png)

The loss curve shows stable convergence under a likelihood-based objective, even in the presence of discontinuities.  
Careful control of learning rate and uncertainty regularization ensures numerical stability and robust optimization.

---

### Normalized Residuals

![Residuals](plots/residuals_discontinuous.png)

Residuals remain approximately centered and symmetric, indicating that both the mean prediction and the learned uncertainty are statistically consistent with the data.

---

## Key Capabilities Demonstrated

ProbNN explicitly addresses several practical challenges in applied machine learning:

- Learning **nonlinear regression functions**  
- Modeling **heteroscedastic uncertainty**  
- Handling **discontinuities and sharp transitions**  
- Maintaining **training stability** under difficult optimization regimes  
- Providing **probabilistic diagnostics**, not just point estimates  

These properties make the framework well suited for scientific modeling, engineering regression, and data-driven systems where uncertainty matters.

---

## Usage

```bash
python main.py
````

This runs all benchmark examples and generates:

* Predictive fits
* Loss curves
* Residual diagnostics

All outputs are saved automatically to `plots/` and `results/`.

---

## Project Structure

```
ProbNN/
├── activation.py     # Activation functions and derivatives
├── nn.py             # Neural network, loss, training, prediction
├── examples.py       # Regression benchmarks
├── utils.py          # Plotting and diagnostics
├── plots/            # Generated figures
├── results/          # Numerical outputs
├── main.py           # Entry point
└── requirements.txt
```

---

## License

MIT License
