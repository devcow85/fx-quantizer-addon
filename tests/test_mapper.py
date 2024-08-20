import pytest
from fx_quantizer_addon import FxNodeMapper
from fx_quantizer_addon import qfunction_set_v1
from fx_quantizer_addon import FxQuantizer
import torch.nn as nn
import torch
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
def quantized_model(model):
    example_input = torch.randn(1, 10)
    quantizer = FxQuantizer(model, example_input)
    data_loader = DataLoader(TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,))), batch_size=10)
    quant_model = quantizer.ptsq(data_loader, qconfig='per_tensor_lwnpu', num_batches=1)
    return quant_model

def test_opset_mapper_generate_scale_table(quantized_model):
    opset_mapper = FxNodeMapper(quantized_model, qfunction_set_v1)
    scale_table = opset_mapper.generate_scale_table()
    assert not scale_table.empty

def test_opset_mapper_map_layers(quantized_model):
    opset_mapper = FxNodeMapper(quantized_model, qfunction_set_v1)
    opset_mapper.generate_scale_table()
    mapped_model = opset_mapper.map_layers()
    assert isinstance(mapped_model, torch.fx.GraphModule)

def test_opset_mapper_find_scale(quantized_model):
    opset_mapper = FxNodeMapper(quantized_model, qfunction_set_v1)
    scale_table = opset_mapper.generate_scale_table()
    for node_name in scale_table.index:
        scale = opset_mapper.find_scale(node_name)
        assert scale is not None
