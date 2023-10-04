import torch

from src.veriflow.linalg import solve_triangular


def test_solve_triangular():
    M_base = torch.ones(10, 10)
    y = torch.arange(10) + 1
    x = torch.ones(10)

    M_lower = torch.tril(M_base)
    x_lower = solve_triangular(M_lower, y)
    assert (x_lower == x).all()

    M_upper = torch.triu(M_base)
    x_upper = solve_triangular(M_upper, y.flip(0))
    assert (x_upper == x).all()
