import pytest
from fx_quantizer_addon import set_seed, sqnr
import numpy as np
import torch

def test_set_seed():
    set_seed(7)
    assert np.random.randint(0, 100) == 47
    assert torch.randint(0, 100, (1,)).item() == 15

def test_sqnr():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.1, 2.1, 3.1])
    sqnr_value = sqnr(x, y)
    assert sqnr_value < 30  # Example threshold, adjust based on expected behavior
