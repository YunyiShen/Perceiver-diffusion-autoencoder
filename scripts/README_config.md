# GALAH Spectra Training Configuration System

This document describes the configuration system for the GALAH spectra training script (`GALAHspectra_train.py`).

## Overview

The training script now uses a JSON configuration file to manage all training parameters, making it easier to:
- Experiment with different hyperparameters
- Reproduce training runs
- Share configurations between team members
- Override specific parameters via command line

## Configuration File Structure

The configuration file (`config.json`) is organized into logical sections:

### Data Configuration
```json
{
    "data": {
        "data_path": "/path/to/spectra/data",
        "model_path": "/path/to/save/models",
        "test_name": "galah_20k"
    }
}
```

### Training Parameters
```json
{
    "training": {
        "epoch": 300,
        "lr": 2.5e-4,
        "batch": 16,
        "save_every": 20,
        "aug": 1
    }
}
```

### Model Architecture
```json
{
    "model": {
        "bottlenecklen": 8,
        "bottleneckdim": 8,
        "model_dim": 256,
        "encoder_layers": 4,
        "decoder_layers": 4,
        "concat": true,
        "cross_attn_only": false,
        "regularize": 0.00
    }
}
```

### Distributed Training
```json
{
    "distributed": {
        "init_method": "tcp://127.0.0.1:23456",
        "backend": "nccl"
    }
}
```

### Data Processing
```json
{
    "data_processing": {
        "threshold_low": -10,
        "threshold_high": 10,
        "num_workers": 1,
        "pin_memory": true
    }
}
```

## Usage Examples

### 1. Using Default Configuration
```bash
# Use the default config.json file
python GALAHspectra_train.py
```

### 2. Using Custom Configuration File
```bash
# Specify a custom configuration file
python GALAHspectra_train.py train --config_path=my_config.json
```

### 3. Overriding Parameters via Command Line
```bash
# Override specific parameters without modifying the config file
python GALAHspectra_train.py train --epoch=100 --lr=1e-4 --batch=32
```

### 4. Combining Custom Config with Overrides
```bash
# Use custom config but override specific parameters
python GALAHspectra_train.py train --config_path=large_model_config.json --epoch=200
```

## Creating Different Configurations

### Example: Small Model for Fast Training
```json
{
    "model": {
        "model_dim": 128,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "bottlenecklen": 4,
        "bottleneckdim": 4
    },
    "training": {
        "epoch": 100,
        "lr": 5e-4
    }
}
```

### Example: Large Model for Better Performance
```json
{
    "model": {
        "model_dim": 512,
        "encoder_layers": 8,
        "decoder_layers": 8,
        "bottlenecklen": 16,
        "bottleneckdim": 16
    },
    "training": {
        "epoch": 500,
        "lr": 1e-4,
        "batch": 8
    }
}
```

## Parameter Descriptions

### Training Parameters
- `epoch`: Number of training epochs
- `lr`: Learning rate for the optimizer
- `batch`: Batch size for training
- `save_every`: Save model checkpoint every N epochs
- `aug`: Data augmentation factor

### Model Parameters
- `bottlenecklen`: Length of the bottleneck sequence
- `bottleneckdim`: Dimension of the bottleneck
- `model_dim`: Hidden dimension of the transformer layers
- `encoder_layers`: Number of encoder layers
- `decoder_layers`: Number of decoder layers
- `concat`: Whether to concatenate features
- `cross_attn_only`: Use only cross-attention in decoder
- `regularize`: Regularization strength

### Data Processing Parameters
- `threshold_low`: Lower threshold for flux normalization
- `threshold_high`: Upper threshold for flux normalization
- `num_workers`: Number of data loading workers
- `pin_memory`: Whether to pin memory for faster GPU transfer

## Validation and Error Handling

The configuration system includes robust error handling:

1. **File Not Found**: If the config file doesn't exist, a clear error message is shown
2. **Invalid JSON**: If the JSON is malformed, the specific error is reported
3. **Unknown Parameters**: If command-line overrides include unknown parameters, a warning is shown
4. **Parameter Validation**: The system validates that all required parameters are present

## Best Practices

1. **Version Control**: Keep your configuration files in version control
2. **Naming Convention**: Use descriptive names for different configs (e.g., `config_small_model.json`)
3. **Documentation**: Add comments to your config files explaining parameter choices
4. **Backup**: Keep backups of working configurations
5. **Experiments**: Create separate config files for different experiments

## Example Scripts

The `example_usage.py` script demonstrates:
- Creating different configuration files
- Validating configurations
- Showing usage patterns

Run it with:
```bash
python example_usage.py
```

## Migration from Old Script

If you were using the old script with hardcoded parameters:

**Old way:**
```python
train(epoch=300, lr=2.5e-4, batch=16, model_dim=256)
```

**New way:**
```bash
# Option 1: Use config file
python GALAHspectra_train.py

# Option 2: Override specific parameters
python GALAHspectra_train.py train --epoch=300 --lr=2.5e-4 --batch=16 --model_dim=256
```

## Troubleshooting

### Common Issues

1. **Config file not found**: Make sure the config file exists in the current directory
2. **Invalid JSON**: Use a JSON validator to check your config file syntax
3. **Unknown parameters**: Check the parameter names in the documentation above
4. **Type errors**: Ensure numeric parameters are numbers, not strings

### Debug Mode

To debug configuration loading, you can add print statements to see what's being loaded:

```python
from GALAHspectra_train import load_config
config = load_config("my_config.json")
print(json.dumps(config, indent=2))
``` 