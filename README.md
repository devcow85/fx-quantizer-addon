# fx-quantizer-addon
FX Quantizer Addon is a custom quantization layer mapping tool designed for PyTorch's FX graph mode. It provides a flexible framework for applying quantization techniques to PyTorch models using FX Graphs, enabling efficient inference on hardware that benefits from quantized models.

## Installation
To install the FX Quantizer Addon, clone the repository and install it using `setup.py`:


```bash
git clone https://github.com/devcow85/fx-quantizer-addon.git
cd fx-quantizer-addon
pip install .
```

## Usage
Here's a basic example of how to use the FX Quantizer Addon:

```python
import torch
from fx_quantizer_addon import FXQuantizer

model = ...  # your PyTorch model
example_input = torch.randn(1, 10)

# Initialize the FX Quantizer with the model and an example input
quantizer = FXQuantizer(model, example_input)

# Apply post-training static quantization (PTSQ) with a custom quantization configuration
quant_model = quantizer.ptsq(data_loader, qconfig='per_tensor_lwnpu', num_batches=1)

# The quantized model is now an FX Graph Module
```

### Module Overview
Each module in the package provides specific functionality:

- `fx_evaluator.py`: Provides tools for evaluating models within the FX Graph mode.
- `fx_quantizer.py`: Contains the main quantization logic, including different quantization strategies.
- `node_mapper.py`: Maps FX nodes to their quantized counterparts.
- `node_tracer.py`: Traces model operations to facilitate the quantization process.
- `qfunctions.py`: Implements quantization-specific functions to simulate hardware behavior.
- `utils.py`: Utility functions to assist with various tasks within the package.

For more detailed usage examples, please refer to the `examples` directory in the repository.


### Test Result
Below are some benchmark results demonstrating the quantization performance using this tool:
| Model   | Dataset          | Quantized Method   | Val Acc @ Top1 | Note       |
|---------|------------------|--------------------|----------------|------------|
| VGG16 | ImageNet2012_v1  | None               | 71.52          | Baseline   |
| VGG16 | ImageNet2012_v1  | per_tensor_lwnpu   | 65.47          | SW Quant   |
| VGG16 | ImageNet2012_v1  | per_tensor_lwnpu   | 64.18          | HW Quant   |
| ResNet18 | ImageNet2012_v1  | None               | 69.68          | Baseline   |
| ResNet18 | ImageNet2012_v1  | per_tensor_lwnpu   | 68.84          | SW Quant   |
| ResNet18 | ImageNet2012_v1  | per_tensor_lwnpu   | 68.78          | HW Quant   |
| ResNet34 | ImageNet2012_v1  | None               | 73.23          | Baseline   |
| ResNet34 | ImageNet2012_v1  | per_tensor_lwnpu   | 72.59          | SW Quant   |
| ResNet34 | ImageNet2012_v1  | per_tensor_lwnpu   | 72.49          | HW Quant   |
| ResNet50 | ImageNet2012_v1  | None               | 80.27          | Baseline   |
| ResNet50 | ImageNet2012_v1  | per_tensor_lwnpu   | 78.45          | SW Quant   |
| ResNet50 | ImageNet2012_v1  | per_tensor_lwnpu   | 77.58          | HW Quant   |


## Running Tests
To run the tests for this package, you can use pytest. Tests are located in the tests directory:

```bash
pytest
# or
pytest tests/
```

This command will execute all the test files in the tests/ directory, ensuring that the package works as expected.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.