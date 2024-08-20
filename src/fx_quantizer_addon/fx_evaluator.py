import torch
from tqdm import tqdm

from .node_tracer import FxNodeTracer

class FxQuantizationEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
        self.model.eval()
        self.model.to(self.device)
        
    def evaluate(self, data_loader):
        total_acc = 0
        total_len = 0

        with torch.inference_mode():
            with tqdm(data_loader, unit="batch") as nbatch:
                for data, targets in nbatch:
                    data, targets = data.to(self.device), targets.to(self.device)
                    outputs = self.model(data)

                    total_acc += (outputs.max(dim=1)[1] == targets).sum().detach().cpu().numpy()
                    total_len += len(data)

                    nbatch.set_postfix_str(f"Val Acc: {total_acc / total_len:.4f}")

        return total_acc / total_len * 100 
        
    def trace_node(self, input_x):
        fqa_tracer = FxNodeTracer()
        fqa_tracer.register_hooks(self.model)
        self.model(input_x.to(self.device))
        
        return fqa_tracer
