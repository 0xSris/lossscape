import numpy as np
from lossscape.surfaces import SURFACES


def test_surface_gradients_are_correct():
    eps = 1e-6

    for surface in SURFACES:
        x = np.array([1.0, -1.0])

        grad_analytic = surface.grad(x)
        grad_numeric = np.zeros_like(x)

        for i in range(2):
            dx = np.zeros_like(x)
            dx[i] = eps
            f_plus = surface.func(x + dx)
            f_minus = surface.func(x - dx)
            grad_numeric[i] = (f_plus - f_minus) / (2 * eps)

        assert np.allclose(
            grad_analytic,
            grad_numeric,
            rtol=1e-3,
            atol=1e-3
        )
