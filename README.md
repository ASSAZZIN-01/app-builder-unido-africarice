# app-builder-unido-africarice

This repo contains tools to export the AfricaRice winning model to ONNX and validate the outputs.

## Setup

1) Create the Python environment:

```
bash install.sh
```

**Important:** The requirements pin `onnx==1.15.0` to ensure exported models use IR version 9, which is compatible with mobile ONNX Runtime. Do not upgrade onnx beyond 1.15.x.

For more details on mobile compatibility, see [MOBILE_COMPATIBILITY.md](MOBILE_COMPATIBILITY.md).

2) Activate your environment (example):

```
source /venv/unido-app-model-export/bin/activate
```

## Download the model checkpoint

The checkpoint is required for ONNX export.

```
cd RiceChallenge_final
python download_checkpoint.py
```

## (Optional) Download data for validation

If you want to run submission scripts and compare outputs, download the images:

```
cd RiceChallenge_final
python download_unido_images.py
```

## Export the model

Use the menu-driven export script from the repo root:

```
bash export_model.sh
```

Options:
- Export ONNX (FP32)
- Export ONNX FP16
- Export ONNX INT8

You can also choose to run a test submission after export.