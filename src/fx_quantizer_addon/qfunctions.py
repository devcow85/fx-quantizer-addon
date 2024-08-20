import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn as aonn

FLOOR_ROUND = 0.5

def int_representator(tensor, k, use_floor = False, rtype = 'float'):
    factor = 2**k
    factor_r = 1
    if rtype == "float":
        factor_r = factor
    
    rounded = (tensor*factor_r).round() if not use_floor else (tensor*factor_r).floor()
    clamped = rounded.clamp(min=-factor, max = factor - 1)
    result = clamped / factor_r
    
    return result

def floor_and_clamp(tensor, bit_depth = 16, signed = True):
    if signed:
        k = 2**(bit_depth-1)
    else:
        k = 2**(bit_depth)
        
    return torch.floor(tensor).clamp(-k, k - 1)

class QConv2dLHW(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, scale=None, input_scale = None, **kwargs):
        qweight = kwargs.pop('qweight', None)
        qbias = kwargs.pop('qbias', None)
        cout_shift = kwargs.pop('out_shift', 0)
        kwargs.pop('zero_point', 0) # remove kwargs list
        
        super().__init__(in_channels, out_channels, kernel_size, padding=padding, stride=stride, **kwargs)
        
        # cout_shift = 3
        
        if qweight is not None:
            assert self.weight.size() == qweight.size(), "custom_weight 크기가 일치하지 않습니다."
            if isinstance(qweight.qscheme(), type(torch.per_tensor_affine)):
                weight_scale = qweight.q_scale()
            else:
                weight_scale = qweight.q_per_channel_scales()
                
            self.weight = nn.Parameter(qweight.int_repr().float())
        
        
        if qbias is not None:
            # assert self.bias.size() == qbias.size(), "custom_bias 크기가 일치하지 않습니다."
            adjusted_bias = qbias / (input_scale * weight_scale) #/ 2**self.cout_shift
            adjusted_bias = floor_and_clamp(adjusted_bias+FLOOR_ROUND, 14)
            self.bias = nn.Parameter(adjusted_bias.unsqueeze(1).unsqueeze(1))
        
        cout_shift = 0 #16-max_value
        self.register_buffer('cout_shift', torch.tensor(cout_shift))
        
        out_scale = torch.tensor(input_scale * weight_scale / scale)*(2**self.cout_shift)
        out_scale = floor_and_clamp(out_scale*(2**15)+FLOOR_ROUND, 16)

        # bn input set (bypass setting)
        self.bn_a = nn.Parameter(torch.ones((self.weight.shape[0],1,1)))
        self.register_buffer('bn_b_lshift', torch.tensor(0))
        self.register_buffer('bn_rshift', torch.tensor(0))
        
        # # quant input set
        self.register_buffer('quant_a', torch.tensor(out_scale))
        self.register_buffer('quant_a_rshift', torch.tensor(0))
        self.register_buffer('quant_b', torch.tensor(2**14-1))  # adjust for round (rep = 0.5/0.4999)
        self.register_buffer('quant_rshift', torch.tensor(15))
        self.register_buffer('conv_output', torch.tensor(0))
        
    def lw_conv(self, x):
        """
        hw conv only use ifmap (x), weight (w) as input with 8bit integer dtype and ofmap(y) has 16bit integer dtype 
        inputs : x, w
        """
        y = F.conv2d(x, self.weight, None, self.stride, self.padding, self.dilation, self.groups)
        y = y / (2**self.cout_shift)
        return floor_and_clamp(y, 16)
    
    def lw_bn(self, x):
        """
        hw bn take ifmap (x), bn_alpha (ba), bn_beta(bb), beta_fp(bs), shift(s) and calculate per-channel mul and add
        > y = (x * bn_a + (self.bias << bn_b_lshift) ) >> bn_rshift
        > y = clip(y , -2^15, 2^15)
        inputs : bn_a, bn_b, bn_b_lshift, bn_rshift
        """
        y = floor_and_clamp(x*self.bn_a, 32)
        y = floor_and_clamp(y+self.bias*(2**self.bn_b_lshift), 32)
        return floor_and_clamp(y/(2**self.bn_rshift), 16)
        
    def lw_quant(self, x):
        """
        hw quant take ifmap (x), q_alpha (qa), q_beta (qb), alpha_shift(qas), beta_shift(qbs) and calculate per-tensor mul and add
        > y = ( ( (x * quant_a) >> quant_a_rshift) + quant_b ) >> quant_rshift
        > y = clip(y, -2^15, 2^15)
        inputs : quant_a, quant_a_rshift, quant_b, quant_rshift
        """
        y = floor_and_clamp((x*self.quant_a)/(2**self.quant_a_rshift), 31)
        y = floor_and_clamp(y + self.quant_b, 32)
        return floor_and_clamp(y/(2**self.quant_rshift), 8)
    
    def forward(self, x):
        y = self.lw_conv(x)
        self.conv_output = y.detach().cpu()
        y = self.lw_bn(y)
        y = self.lw_quant(y)
        return y
        
class QConvReLU2dLHW(QConv2dLHW):
    def forward(self, x):
        y = super().forward(x)
        return F.relu(y)

class QLinearLHW(nn.Linear):
    def __init__(self, in_features, out_features, scale=None, input_scale = None, **kwargs):
        qweight = kwargs.pop('qweight', None)
        qbias = kwargs.pop('qbias', None)
        lout_shift = kwargs.pop('out_shift', 0)
        kwargs.pop('zero_point', 0) # remove kwargs list
        kwargs.pop('qscheme', 0) # remove kwargs list
        
        super().__init__(in_features, out_features, **kwargs)

        # lout_shift = 3
        self.register_buffer('lout_shift', torch.tensor(lout_shift))
        
        if qweight is not None:
            assert self.weight.size() == qweight.size(), "custom_weight 크기가 일치하지 않습니다."
            if isinstance(qweight.qscheme(), type(torch.per_tensor_affine)):
                weight_scale = qweight.q_scale()
            else:
                weight_scale = qweight.q_per_channel_scales()
                
            self.weight = nn.Parameter(qweight.int_repr().float())
        
        if qbias is not None:
            assert self.bias.size() == qbias.size(), "custom_bias 크기가 일치하지 않습니다."
            adjusted_bias = qbias / (input_scale * weight_scale) #/ 2**self.cout_shift
            adjusted_bias = floor_and_clamp(adjusted_bias+FLOOR_ROUND, 14)
            self.bias = nn.Parameter(adjusted_bias.unsqueeze(0))
            
        out_scale = torch.tensor(input_scale * weight_scale / scale)*(2**self.lout_shift)
        out_scale = floor_and_clamp(out_scale*(2**15)+FLOOR_ROUND, 16)

        # bn input set (bypass setting)
        self.bn_a = nn.Parameter(torch.ones((1, self.weight.shape[0])))
        self.register_buffer('bn_b_lshift', torch.tensor(0))  # (2**k)
        self.register_buffer('bn_rshift', torch.tensor(0))    # (2**k)
        
        # quant input set (bypass setting)
        self.register_buffer('quant_a', out_scale)
        self.register_buffer('quant_a_rshift', torch.tensor(0))   # (2**k)
        self.register_buffer('quant_b', torch.tensor(2**14-1))
        self.register_buffer('quant_rshift', torch.tensor(15))     # (2**k)
        self.register_buffer('linear_output', torch.tensor(0))
        
            
    def lw_linear(self, x):
        """
        hw linear only use ifmap (x), weight (w) as input with 8bit integer dtype and ofmap(y) has 16bit integer dtype 
        inputs : x, w
        """
        y = F.linear(torch.floor(x), self.weight, None)
        y = y / (2**self.lout_shift)
        return floor_and_clamp(y, 16)
    
    def lw_bn(self, x):
        """
        hw bn take ifmap (x), bn_alpha (ba), bn_beta(bb), beta_fp(bs), shift(s) and calculate per-channel mul and add
        > y = (x * bn_a + (self.bias << bn_b_lshift) ) >> bn_rshift
        > y = clip(y , -2^15, 2^15)
        inputs : bn_a, bn_b, bn_b_lshift, bn_rshift
        """
        y = floor_and_clamp(x*self.bn_a, 32)
        y = floor_and_clamp(y+self.bias*(2**self.bn_b_lshift), 32)
        return floor_and_clamp(y/(2**self.bn_rshift), 16)
    
    def lw_quant(self, x):
        """
        hw quant take ifmap (x), q_alpha (qa), q_beta (qb), alpha_shift(qas), beta_shift(qbs) and calculate per-tensor mul and add
        > y = ( ( (x * quant_a) >> quant_a_rshift) + quant_b ) >> quant_rshift
        > y = clip(y, -2^15, 2^15)
        inputs : quant_a, quant_a_rshift, quant_b, quant_rshift
        """
        y = floor_and_clamp((x*self.quant_a)/(2**self.quant_a_rshift), 31)
        y = floor_and_clamp(y + self.quant_b, 32)
        return floor_and_clamp(y/(2**self.quant_rshift), 8)
    
    def forward(self, x):
        y = self.lw_linear(x)
        self.linear_output = y.detach().cpu()
        y = self.lw_bn(y)
        y = self.lw_quant(y)
        return y

class QLinearReLULHW(QLinearLHW):
    def forward(self, x):
        y = super().forward(x)
        return F.relu(y)
    
class QuantizeLHW(aonn.quantized.modules.Quantize):
    def forward(self, x):
        y = super(QuantizeLHW, self).forward(x)
        return y.int_repr().to(torch.float) - self.zero_point
        

def add_relu_float(x1, x2, scale, zero_point, scale_x1, scale_x2):
    y = torch.round((x1*scale_x1 + x2*scale_x2)/scale)
    y = F.relu(y)
    return y.clamp(-2**7, 2**7 - 1)

def add_float(x1, x2, scale_x1, scale_x2, scale, zero_point):
    y = torch.round((x1*scale_x1 + x2*scale_x2)/scale)
    return y.clamp(-2**7, 2**7 - 1)
    
def cat_float(x_list, scale_list, scale, zero_point, dim):
    # Step 1: Dequantize all input tensors
    dequantized_tensors = [
        x * scale for x, scale in zip(x_list, scale_list)
    ]

    # Step 2: Concatenate the dequantized tensors
    concatenated_dequant = torch.cat(dequantized_tensors, dim=dim)

    # Step 3: Quantize the concatenated result
    quantized_output = torch.round(concatenated_dequant / scale) + zero_point

    # Step 4: Clamp the result to the range of int8
    quantized_output = quantized_output.clamp(-2**7, 2**7 - 1)

    return quantized_output
    
def quantize_per_tensor_modify(x, input_scale, input_zeropoint, dtype):
    y = torch.quantize_per_tensor(x, input_scale, input_zeropoint, dtype)
    return y.int_repr().float()-input_zeropoint

def parse_extra_repr(extra_repr_str):
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

qfunction_set_v1 = {
    aonn.quantized.modules.conv.Conv2d: QConv2dLHW,
    aonn.intrinsic.quantized.modules.conv_relu.ConvReLU2d: QConvReLU2dLHW,
    aonn.intrinsic.quantized.modules.linear_relu.LinearReLU: QLinearReLULHW,
    aonn.quantized.modules.linear.Linear: QLinearLHW,
    torch.quantize_per_tensor: quantize_per_tensor_modify,
    torch.ops.quantized.add_relu: add_relu_float,
    torch.ops.quantized.add: add_float,
    aonn.quantized.modules.Quantize: QuantizeLHW,
    torch.ops.quantized.cat: cat_float
}