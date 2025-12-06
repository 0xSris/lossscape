import numpy as np
from lossscape.optimizers import SGD, Momentum, AdamLite


def test_sgd_update():
    x = np.array([1.0, 2.0])
    grad = np.array([0.1, -0.2])

    opt = SGD(lr=0.5)
    x_next = opt.step(x, grad)

    expected = x - 0.5 * grad
    np.testing.assert_allclose(x_next, expected)


def test_momentum_moves():
    x = np.array([1.0, 1.0])
    grad = np.array([1.0, 1.0])

    opt = Momentum(lr=0.1, momentum=0.9)
    x1 = opt.step(x, grad)
    x2 = opt.step(x1, grad)

    assert not np.allclose(x1, x2)


def test_adam_step_shape():
    x = np.array([1.0, 2.0])
    grad = np.array([0.3, -0.3])

    opt = AdamLite(lr=0.1)
    x_next = opt.step(x, grad)

    assert x.shape == x_next.shape
