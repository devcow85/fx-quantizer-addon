import pytest
from fx_quantizer_addon import FxQuantizationEvaluator
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)

@pytest.fixture
def model():
    return SimpleModel()

@pytest.fixture
def data_loader():
    x = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=10)

def test_evaluate(model, data_loader):
    evaluator = FxQuantizationEvaluator(model, "cpu")
    accuracy = evaluator.evaluate(data_loader)
    assert 0 <= accuracy <= 100
