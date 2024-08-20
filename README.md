# fx-quantizer-addon
custom quantization layer mapping tools for torch fx graph mode

## Installation
You can install the FX Quantizer Addon package by cloning the repository and using `setup.py`:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install .
```

## Usage
Here's a basic example of how to use the FX Quantizer Addon:

```python
from fx_quantizer_addon import FXQuantizer


model = ...  # your model
example_input = torch.randn(1, 10)

# Example of how to use the fx_quantizer
quantizer = FxQuantizer(model, example_input)
quant_model = quantizer.ptsq(data_loader, qconfig='per_tensor_lwnpu', num_batches=1)
# this quant_model is fx_graph_module
```

Each module in the package provides specific functionality:

- `fx_evaluator.py`: Tools for evaluating models in FX Graph mode.
- `fx_quantizer.py`: The main quantization logic.
- `node_mapper.py`: Maps FX nodes to quantized versions.
- `node_tracer.py`: Traces model operations to facilitate quantization.
- `qfunctions.py`: Quantization-specific custom functions for simulating functional HW behavior.
- `utils.py`: Utility functions for the package.

## Running Tests
To run the tests for this package, you can use pytest. Tests are located in the tests directory:

```shell
pytest
or
pytest tests/
```

This command will execute all the test files in the tests/ directory, ensuring that the package works as expected.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.