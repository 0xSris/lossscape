# src/lossscape/surfaces.py
import numpy as np

class LossSurface:
    """Represents a 2D loss surface with its gradient."""
    def __init__(self, name, func, grad):
        self.name = name
        self.func = func
        self.grad = grad

# Quadratic (simple convex)
def quad(xy):
    x, y = xy
    return 0.5 * (x**2 + 4*y**2)

def grad_quad(xy):
    x, y = xy
    return np.array([x, 4*y])

# Saddle (unstable)
def saddle(xy):
    x, y = xy
    return 0.5 * (x**2 - y**2)

def grad_saddle(xy):
    x, y = xy
    return np.array([x, -y])

# Multimodal (many local minima)
def multimodal(xy):
    x, y = xy
    return np.sin(3*x)*np.sin(3*y) + 0.1*(x**2 + y**2)

def grad_multimodal(xy):
    x, y = xy
    gx = 3*np.cos(3*x)*np.sin(3*y) + 0.2*x
    gy = 3*np.sin(3*x)*np.cos(3*y) + 0.2*y
    return np.array([gx, gy])

# List of available surfaces
SURFACES = [
    LossSurface('Quadratic (easy)', quad, grad_quad),
    LossSurface('Saddle (unstable)', saddle, grad_saddle),
    LossSurface('Multimodal (many minima)', multimodal, grad_multimodal),
]
