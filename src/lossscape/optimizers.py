# src/lossscape/optimizers.py
import numpy as np

class Optimizer:
    """Base optimizer interface."""
    def step(self, x, grad):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr=0.1):
        self.lr = lr

    def step(self, x, grad):
        # Clip gradient to prevent overflow
        grad_clipped = np.clip(grad, -1.0, 1.0)
        return x - self.lr * grad_clipped

class Momentum(Optimizer):
    def __init__(self, lr=0.05, momentum=0.9):
        self.lr = lr
        self.m = momentum
        self.v = np.zeros(2)

    def step(self, x, grad):
        grad_clipped = np.clip(grad, -1.0, 1.0)
        self.v = self.m * self.v + (1 - self.m) * grad_clipped
        return x - self.lr * self.v

class AdamLite(Optimizer):
    def __init__(self, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.m = np.zeros(2)
        self.v = np.zeros(2)
        self.t = 0

    def step(self, x, grad):
        self.t += 1
        grad_clipped = np.clip(grad, -1.0, 1.0)
        self.m = self.b1 * self.m + (1 - self.b1) * grad_clipped
        self.v = self.b2 * self.v + (1 - self.b2) * (grad_clipped ** 2)
        m_hat = self.m / (1 - self.b1 ** self.t + 1e-16)
        v_hat = self.v / (1 - self.b2 ** self.t + 1e-16)
        return x - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# Dictionary to easily access optimizers by name
OPTIMIZERS = {
    "SGD": SGD,
    "Momentum": Momentum,
    "AdamLite": AdamLite,
}
