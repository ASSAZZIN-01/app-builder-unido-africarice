# Mobile ONNX Compatibility Guide

## The Problem

If your mobile app shows an error like:
```
ONNX model IR version 10 is not supported. Maximum supported IR version is 9.
```

This means the ONNX model was exported with a newer ONNX format than your mobile runtime supports.

## The Solution

### For the Model Export Team

1. **Install the correct ONNX version:**
   ```bash
   pip install onnx==1.15.0
   ```

2. **Re-export the model:**
   ```bash
   bash export_model.sh
   ```
   Choose option 1 (FP32) for best compatibility.

3. **Verify the model files:**
   - `ultimate_tiled_multitask.onnx` (main model graph)
   - `ultimate_tiled_multitask.onnx.data` (weights)
   
   Both files must be sent together.

### Why This Happens

- **IR Version 10** was introduced in ONNX Python package version 1.16.0
- **IR Version 9** is the standard for ONNX Python package versions 1.14.0 and 1.15.0
- Mobile ONNX Runtime typically supports up to IR version 9

### Verification

After re-exporting with onnx==1.15.0, you can verify the IR version:

```python
import onnx
model = onnx.load("ultimate_tiled_multitask.onnx")
print(f"IR version: {model.ir_version}")  # Should be 9
```

### Communication Template

> "The ONNX model has been re-exported with IR version 9 for mobile compatibility. Please find attached:
> - `ultimate_tiled_multitask.onnx`
> - `ultimate_tiled_multitask.onnx.data`
> 
> Keep both files in the same directory. The model expects:
> - Input `tiles`: `(1, 48, 3, 512, 512)` float32, RGB in range 0-255
> - Input `meta`: `(1, 3)` float32, one-hot for rice type [Paddy, White, Brown]
> - Outputs: `counts` (9 values) and `measures` (6 values)"

## Requirements Pinned

This repository pins `onnx==1.15.0` in `requirements.txt` to ensure mobile compatibility. Do not upgrade beyond 1.15.x unless your mobile runtime explicitly supports IR version 10+.
