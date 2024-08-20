import re

import torch
import numpy as np
import pandas as pd

def _parse_extra_repr(extra_repr_str):
    # Split the string by commas while keeping the text inside parentheses together
    parts = re.split(r',\s*(?![^()]*\))', extra_repr_str)
    
    args = []
    kwargs = {}
    
    for part in parts:
        if '=' in part:
            key, value = part.split('=', 1)
            key = key.strip()
            value = value.strip()
            try:
                kwargs[key] = eval(value)
            except NameError:
                kwargs[key] = value  # For strings and unrecognized types
        else:
            try:
                args.append(eval(part.strip()))
            except NameError:
                args.append(part.strip())
    
    return args, kwargs


class FxNodeMapper:
    def __init__(self, model, opset):
        self.model = torch.fx.symbolic_trace(model)
        self.opset = opset
        self.scale_table = None
        self.modified_traced = None
        
        self.generate_scale_table()

    def generate_scale_table(self):
        
        scale_dict = {'node_name': [], 'node_op_type': [], 'scale': [], 'zero_point': [], 'input_nodes': []}
        
        for node in self.model.graph.nodes:
            if node.op == 'call_module':
                module = self.model.get_submodule(node.target)
                scale = module.scale if hasattr(module, 'scale') else None
                zero_point = module.zero_point if hasattr(module, 'zero_point') else None

                scale_dict['node_name'].append(node.name)
                scale_dict['scale'].append(scale.item() if isinstance(scale, torch.Tensor) else scale)
                scale_dict['zero_point'].append(zero_point.item() if isinstance(zero_point, torch.Tensor) else zero_point)
                scale_dict['input_nodes'].append([r.name for r in node.all_input_nodes])
                scale_dict['node_op_type'].append(type(module).__name__)

            elif node.op in ('call_function', 'call_method'):
                scale, zero_point = self._extract_scale_zero_point(node)
                scale_dict['node_name'].append(node.name)
                scale_dict['scale'].append(scale.item() if isinstance(scale, torch.Tensor) else scale)
                scale_dict['zero_point'].append(zero_point.item() if isinstance(zero_point, torch.Tensor) else zero_point)
                scale_dict['input_nodes'].append([r.name for r in node.all_input_nodes])
                scale_dict['node_op_type'].append(node.target.__name__ if node.op == "call_function" else node.target)

        self.scale_table = pd.DataFrame(scale_dict).replace(np.nan, None).set_index("node_name")
        return self.scale_table

    def _extract_scale_zero_point(self, node):
        scale, zero_point = None, None
        for arg in node.args:
            if hasattr(arg, 'target'):
                if isinstance(arg.target, str):
                    if arg.op == "get_attr" and 'scale' in arg.target:
                        scale = getattr(self.model, arg.target).detach().cpu().float()
                    if arg.op == "get_attr" and 'zero_point' in arg.target:
                        zero_point = getattr(self.model, arg.target).detach().cpu().float()
        return scale, zero_point

    def find_scale(self, node_name):
        scale = self.scale_table.loc[node_name]['scale']
        if scale is not None:
            return scale
        else:
            input_nodes = self.scale_table.loc[node_name]['input_nodes']
            for inode in input_nodes:
                scale = self.find_scale(inode)
                if scale is not None:
                    return scale
            return None

    def map_layers(self):
        for nidx, node in enumerate(self.model.graph.nodes):
            if node.op == "call_module":
                self._replace_call_module(node, nidx)
            elif node.op in ('call_function', 'call_method'):
                self._handle_call_function_or_method(node, nidx)

        self.modified_traced = torch.fx.GraphModule(self.model, self.model.graph)
        return self.modified_traced

    def _replace_call_module(self, node, nidx):
        target_layer = self.model.get_submodule(node.target)
        input_nodes = self.scale_table.loc[node.name]["input_nodes"]

        if type(target_layer) in self.opset:
            # out_scale = self.scale_table.loc[node.name]["scale"]
                
            if isinstance(target_layer, torch.ao.nn.quantized.modules.Quantize):
                args, kwargs = _parse_extra_repr(target_layer.extra_repr())
                replace_layer = self.opset[type(target_layer)](scale = target_layer.scale, zero_point = target_layer.zero_point, dtype = target_layer.dtype)
                self.model.add_submodule(node.target, replace_layer)
                
            else:
                input_scale = [self.find_scale(inode) for inode in input_nodes]
                args, kwargs = _parse_extra_repr(target_layer.extra_repr())
                params = {"qweight": target_layer.weight().data, "qbias": torch.zeros(target_layer.weight().shape[0]) if target_layer.bias() is None else target_layer.bias().data}
                
                if nidx > 5:
                    params["out_shift"] = 3
                
                replace_layer = self.opset[type(target_layer)](*args, **kwargs, input_scale=input_scale[0], **params)
                self.model.add_submodule(node.target, replace_layer)

    def _handle_call_function_or_method(self, node, nidx):
        if (node.target == torch.ops.quantized.add_relu) or (node.target == torch.ops.quantized.add):
            input_nodes = self.scale_table.loc[node.name]["input_nodes"][:2]
            input_scale = [self.find_scale(inode) for inode in input_nodes]
            node.args = tuple(list(node.args) + input_scale)

        if (node.target == torch.ops.quantized.cat):
            input_nodes = self.scale_table.loc[node.name]["input_nodes"]
            input_scale = [self.find_scale(inode) for inode in input_nodes]
            node.args += (input_scale, )
        
        if node.target in self.opset:
            node.target = self.opset[node.target]
            node.args = tuple(list(node.args))

    def get_modified_traced_model(self):
        return self.modified_traced