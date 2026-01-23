import numpy as np


class ReLU:
    """Rectified Linear Unit activation."""

    def forward(self, z):
        self.z = z  # cache pre-activation
        return np.maximum(0.0, z)

    def backward(self, da):
        # derivative: 1 for z > 0, 0 otherwise
        return da * (self.z > 0.0)


class Tanh:
    """Hyperbolic tangent activation."""

    def forward(self, z):
        self.a = np.tanh(z)  # cache activation
        return self.a

    def backward(self, da):
        # derivative: 1 - tanh(z)^2
        return da * (1.0 - self.a ** 2)


class Sigmoid:
    """Logistic sigmoid activation."""

    def forward(self, z):
        self.a = 1.0 / (1.0 + np.exp(-z))  # cache activation
        return self.a

    def backward(self, da):
        # derivative: a * (1 - a)
        return da * self.a * (1.0 - self.a)
