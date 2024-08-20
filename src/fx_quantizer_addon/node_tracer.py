import pickle

import torch

# tracing helper function
def _tensor_to_numpy(tensor):
    if tensor.is_quantized:
        tensor = tensor.int_repr().float()
    return tensor.detach().cpu().numpy()

def _convert_state_dict_to_numpy(state_dict):
    numpy_state_dict = {}
    for key, value in state_dict.items():
        numpy_state_dict[key] = _tensor_to_numpy(value) if isinstance(value, torch.Tensor) else value
    return numpy_state_dict

# node trace class
class FxNodeTracer:
    def __init__(self):
        self.trace_dict = {}
    
    def trace_node(self, node, trace_dict):
        def hook(module, input, output):
            state_dict = module.state_dict()
            extra_repr = module.extra_repr()
            trace_dict[node.name] = {
                "node_type": type(module).__name__,
                "input": tuple(_tensor_to_numpy(tensor) if isinstance(tensor, torch.Tensor) else tensor for tensor in input),
                "output": _tensor_to_numpy(output),
                "state_dict": _convert_state_dict_to_numpy(state_dict),
                "extra_repr": extra_repr,
                "input_nodes": [inode.name for inode in node._input_nodes]
            }
        return hook
    
    def trace_function(self, node, trace_dict):
        def wrapper(*args, **kwargs):
            output = node.target(*args, **kwargs)
            trace_dict[node.name] = {
                "node_type": node.target.__name__,
                "input": tuple(_tensor_to_numpy(tensor) if isinstance(tensor, torch.Tensor) else tensor for tensor in args),
                "output": _tensor_to_numpy(output),
                "input_nodes": [inode.name for inode in node._input_nodes]
            }
            return output
        return wrapper
        
    def register_hooks(self, model):
        for node in model.graph.nodes:
            if node.op == 'call_module':
                module = model.get_submodule(node.target)
                module.register_forward_hook(self.trace_node(node, self.trace_dict))
            elif node.op == 'call_function':
                if not node.name.__contains__('getitem'):
                    wrapped_func = self.trace_function(node, self.trace_dict)
                    with model.graph.inserting_before(node):
                        new_node = model.graph.call_function(wrapped_func, node.args, node.kwargs)
                        node.replace_all_uses_with(new_node)
                        model.graph.erase_node(node)
        model.recompile()
        
        return model

    def save_trace_data(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.trace_dict, f)