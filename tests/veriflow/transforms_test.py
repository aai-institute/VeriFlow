import torch

from src.veriflow.transforms import (LeakyReLUTransform, LUTransform, Permute,
                                     ScaleTransform)


def test_scale_transform():
    """Test scale transform."""
    # Test input
    dim = 10
    transform = ScaleTransform(10)
    with torch.no_grad():
        transform.scale.copy_(torch.ones(dim) * 2)
    x = torch.ones(dim)

    # Test forward, inverse, and log det
    y = transform(x)
    assert (y == 2 * x).all()
    assert (transform.backward(y) == x).all()
    log_det = transform.log_abs_det_jacobian(x, y)
    assert log_det == dim * torch.log(torch.tensor(2))


def test_permute():
    """Test permute."""
    # Test input
    dim = 10
    transform = Permute(torch.arange(dim))
    x = torch.arange(dim)

    # Test forward, inverse, and log det
    y = transform(x)
    assert (y == x).all()
    assert (transform._inverse(y) == x).all()
    log_det = transform.log_abs_det_jacobian(x, y)
    assert log_det == 0


def test_lu_transform():
    """Test LU transform."""
    # Test input
    dim = 10
    transform = LUTransform(dim)
    with torch.no_grad():
        transform.L_raw.copy_(torch.tril(torch.ones(dim, dim)))
        transform.U_raw.copy_(torch.eye(dim))
        transform.bias.copy_(torch.zeros(dim))
    x = torch.ones(dim)

    # Test forward, inverse, and log det
    y = transform.backward(x)  # LU-factorization parametrizes inverse
    assert (y == (torch.arange(dim) + 1.0)).all()
    assert (transform(y) == x).all()
    log_det = transform.log_abs_det_jacobian(x, y)
    assert log_det == 0


def test_leaky_relu_transform():
    """Test leaky ReLU transform."""
    # Test input
    base_dim = 5
    dim = 2 * base_dim
    transform = LeakyReLUTransform()
    x = torch.tensor([1.0, -1.0] * base_dim)
    y_true = x * torch.tensor([1.0, 0.01] * base_dim)

    # Test forward, inverse, and log det
    y = transform(x)
    assert (y == y_true).all()
    assert (transform.backward(y) == x).all()
    log_det = transform.log_abs_det_jacobian(x, y)
    assert log_det == base_dim * torch.log(torch.tensor(0.01))
