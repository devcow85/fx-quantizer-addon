import torch
import torch.fx as fx
import torch.quantization
import torch.quantization.quantize_fx as quantize_fx
from tqdm import tqdm

qconfig_preset = {
    'fbgemm': torch.quantization.get_default_qconfig('fbgemm'),
    'x86': torch.quantization.get_default_qconfig('x86'),
    'per_tensor_lwnpu': torch.quantization.QConfig(activation=torch.quantization.HistogramObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.quint8, quant_max=127, quant_min=0), 
                                                   weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)),
    'per_channel_lwnpu': torch.quantization.QConfig(activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine), 
                                                    weight=torch.quantization.PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
}

class FxQuantizer:
    def __init__(self, model, example_input): 
        self.model = model
        self.example_input = example_input
        
    def prepare_model(self, qconfig):
        self.model.eval()
        model_fx = fx.symbolic_trace(self.model)
        prepared_model = quantize_fx.prepare_fx(model_fx, 
                                                torch.ao.quantization.QConfigMapping().set_global(qconfig),
                                                example_inputs=(self.example_input,))
        prepared_model(self.example_input)
        return prepared_model
    
    def convert_model(self, prepared_model):
        prepared_model.to("cpu")
        return quantize_fx.convert_fx(prepared_model)
    
    def calibration(self, data_loader, num_batches=None, device="cuda"):
        self.model.eval()
        self.model.to(device)
        with torch.no_grad():
            for i, (data, _) in enumerate(tqdm(data_loader, desc='Calibration')):
                if num_batches is not None and i >= num_batches:
                    break
                self.model(data.to(device))

    def train(self, data_loader, optimizer, criterion, num_epochs=5):
        self.model.train()
        for epoch in range(num_epochs):
            for data, target in tqdm(data_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
    
    def ptsq(self, data_loader, qconfig='per_tensor_lwnpu', num_batches=None):
        if isinstance(qconfig, str):
            qconfig = qconfig_preset[qconfig]

        prepared_model = self.prepare_model(qconfig)
        self.calibration(data_loader, num_batches)
        return self.convert_model(prepared_model)
    
    def ptdq(self, qconfig='per_tensor_lwnpu'):
        if isinstance(qconfig, str):
            qconfig = qconfig_preset[qconfig]
        
        prepared_model = self.prepare_model(qconfig)
        return self.convert_model(prepared_model)
    
    def qat(self, data_loader, optimizer, criterion, num_epochs=5, qconfig='per_tensor_lwnpu'):
        if isinstance(qconfig, str):
            qconfig = qconfig_preset[qconfig]
            
        self.model.train()
        self.model.qconfig = qconfig
        prepared_model = torch.quantization.prepare_qat(self.model, inplace=False)
        self.train(data_loader, optimizer, criterion, num_epochs)
        return torch.quantization.convert(prepared_model)
