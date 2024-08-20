import pytest
from fx_quantizer_addon import FxNodeTracer
import torch.nn as nn
import torch

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

@pytest.fixture
def model():
    return SimpleModel()

def test_node_tracer_register_hooks(model):
    tracer = FxNodeTracer()
    traced = torch.fx.symbolic_trace(model)
    tracer.register_hooks(traced)
    input_tensor = torch.randn(1, 10)
    traced(input_tensor)
    assert len(tracer.trace_dict) > 0

def test_node_tracer_trace_node(model):
    tracer = FxNodeTracer()
    traced = torch.fx.symbolic_trace(model)
    tracer.register_hooks(traced)
    input_tensor = torch.randn(1, 10)
    traced(input_tensor)
    for node in traced.graph.nodes:
        if node.op == 'call_module':
            assert node.name in tracer.trace_dict
            assert 'node_type' in tracer.trace_dict[node.name]
            assert 'input' in tracer.trace_dict[node.name]
            assert 'output' in tracer.trace_dict[node.name]
            assert 'state_dict' in tracer.trace_dict[node.name]
            assert 'extra_repr' in tracer.trace_dict[node.name]

def test_node_tracer_save_trace_data(model, tmp_path):
    tracer = FxNodeTracer()
    traced = torch.fx.symbolic_trace(model)
    tracer.register_hooks(traced)
    input_tensor = torch.randn(1, 10)
    traced(input_tensor)
    trace_file = tmp_path / "trace.pkl"
    tracer.save_trace_data(trace_file)
    assert trace_file.exists()
