import pytest
import torch
import torch.nn as nn
import torch.ao.nn as aonn
from fx_quantizer_addon import FxNodeMapper
from fx_quantizer_addon import FxQuantizer
from fx_quantizer_addon.qfunctions import (
    QConv2dLHW,
    QConvReLU2dLHW,
    QLinearLHW,
    QLinearReLULHW,
    add_relu_float,
    quantize_per_tensor_modify,
    parse_extra_repr,
    qfunction_set_v1,
)
from torch.utils.data import DataLoader, TensorDataset


torch.set_printoptions(sci_mode=False, precision=4)
ATOL = 5e-2  # 공동 변수로 설정

class SimpleConvModel(nn.Module):
    def __init__(self):
        super(SimpleConvModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

@pytest.fixture
def data_loader():
    x = torch.randn(100, 3, 8, 8)
    y = torch.randint(0, 2, (100,))
    dataset = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(dataset, batch_size=10)

@pytest.fixture
def quantized_model():
    model = SimpleConvModel()
    example_input = torch.randn(1, 3, 8, 8)
    quantizer = FxQuantizer(model, example_input)
    data_loader = DataLoader(TensorDataset(torch.randn(100, 3, 8, 8), torch.randint(0, 2, (100,))), batch_size=10)
    quant_model = quantizer.ptsq(data_loader, qconfig='per_tensor_lwnpu', num_batches=1)
    return quant_model

# def test_qconv2d_mapper(quantized_model, data_loader):
#     # Mapping
#     opset_mapper = FxNodeMapper(quantized_model, fset_1)
#     scale_table = opset_mapper.generate_scale_table()
#     mapped_model = opset_mapper.map_layers()

#     # Testing the quantized model
#     quantized_model.eval()
#     mapped_model.eval()

#     input_tensor = torch.randn(1, 3, 8, 8)

#     with torch.no_grad():
#         quantized_output = quantized_model(input_tensor)
#         mapped_output = mapped_model(input_tensor)

#     scale = quantized_model.conv2.scale
    
#     print(mapped_output[0], scale)
#     print(quantized_output[0])
        
#     assert torch.allclose(mapped_output*scale, quantized_output, atol=ATOL)

# Test other layers similarly if necessary
# Example for QLinear:
def test_qlinear_mapper(quantized_model, data_loader):
    # Similar to the above test_qconv2d_mapper function
    pass

def test_add_relu_float():
    x1 = torch.tensor([0.1, 0.5, 0.9])
    x2 = torch.tensor([0.2, 0.3, 0.4])
    result = add_relu_float(x1, x2, scale=0.1, zero_point=0, scale_x1=1.0, scale_x2=1.0)
    expected = torch.round((x1 * 1.0 + x2 * 1.0) / 0.1).clamp(min=0)
    assert torch.allclose(result, expected, atol=ATOL)

def test_quantize_per_tensor_modify():
    x = torch.tensor([0.1, 0.5, 0.9])
    result = quantize_per_tensor_modify(x, input_scale=0.1, input_zeropoint=0, dtype=torch.qint8)
    expected = torch.round(x / 0.1)
    assert torch.allclose(result, expected, atol=ATOL)

def test_parse_extra_repr():
    extra_repr_str = "3, 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)"
    args, kwargs = parse_extra_repr(extra_repr_str)
    assert args == [3, 6]
    assert kwargs == {
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (1, 1),
    }

def test_qfset_v1_mapping():
    assert qfunction_set_v1[aonn.quantized.modules.conv.Conv2d] == QConv2dLHW
    assert qfunction_set_v1[aonn.intrinsic.quantized.modules.conv_relu.ConvReLU2d] == QConvReLU2dLHW
    assert qfunction_set_v1[aonn.intrinsic.quantized.modules.linear_relu.LinearReLU] == QLinearReLULHW
    assert qfunction_set_v1[aonn.quantized.modules.linear.Linear] == QLinearLHW
    assert qfunction_set_v1[torch.quantize_per_tensor] == quantize_per_tensor_modify
    assert qfunction_set_v1[torch.ops.quantized.add_relu] == add_relu_float
