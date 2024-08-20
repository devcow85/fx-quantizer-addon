import pytest
from fx_quantizer_addon import FxQuantizer, qconfig_preset
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn

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

def test_ptsq(model, data_loader):
    example_input = torch.randn(1, 10)
    quantizer = FxQuantizer(model, example_input)
    quant_model = quantizer.ptsq(data_loader, qconfig='per_tensor_lwnpu', num_batches=1)
    assert isinstance(quant_model, torch.fx.GraphModule)

def test_ptdq(model):
    example_input = torch.randn(1, 10)
    quantizer = FxQuantizer(model, example_input)
    quant_model = quantizer.ptdq(qconfig='per_tensor_lwnpu')
    assert isinstance(quant_model, torch.fx.GraphModule)

def test_qat(model, data_loader):
    example_input = torch.randn(1, 10)
    quantizer = FxQuantizer(model, example_input)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    quant_model = quantizer.qat(data_loader, optimizer, criterion, num_epochs=1, qconfig='per_tensor_lwnpu')
    assert isinstance(quant_model, torch.nn.Module)

def test_calibration(model, data_loader):
    example_input = torch.randn(1, 10)
    quantizer = FxQuantizer(model, example_input)
    qconfig = qconfig_preset['per_tensor_lwnpu']
    prepared_model = quantizer.prepare_model(qconfig)
    
    # Check observer's initial state
    initial_min_max = {}
    for name, module in prepared_model.named_modules():
        if hasattr(module, 'activation_post_process'):
            observer = module.activation_post_process
            initial_min_max[name] = (observer.min_val.clone(), observer.max_val.clone())
    
    quantizer.calibration(data_loader, num_batches=1)
    
    # Check observer's state after calibration
    for name, module in prepared_model.named_modules():
        if hasattr(module, 'activation_post_process'):
            observer = module.activation_post_process
            initial_min, initial_max = initial_min_max[name]
            assert not torch.equal(observer.min_val, initial_min), f"Observer min_val did not change for module {name}"
            assert not torch.equal(observer.max_val, initial_max), f"Observer max_val did not change for module {name}"

def test_prepare_and_convert_model(model):
    example_input = torch.randn(1, 10)
    quantizer = FxQuantizer(model, example_input)
    prepared_model = quantizer.prepare_model(qconfig_preset['fbgemm'])
    quant_model = quantizer.convert_model(prepared_model)
    assert isinstance(quant_model, torch.nn.Module)
