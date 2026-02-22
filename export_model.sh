#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$ROOT_DIR/RiceChallenge_final"

if [[ ! -d "$WORK_DIR" ]]; then
  echo "RiceChallenge_final not found at: $WORK_DIR"
  exit 1
fi

cd "$WORK_DIR"

# Check ONNX version for mobile compatibility
onnx_version=$(python -c "import onnx; print(onnx.__version__)" 2>/dev/null || echo "unknown")
if [[ "$onnx_version" != 1.15.* ]] && [[ "$onnx_version" != 1.14.* ]]; then
  echo "WARNING: onnx version is $onnx_version"
  echo "For mobile compatibility (IR version 9), onnx 1.15.x or 1.14.x is recommended."
  echo "The current version may produce IR version 10 models that won't load on mobile."
  read -r -p "Continue anyway? [y/N]: " proceed
  if [[ "$proceed" != "y" && "$proceed" != "Y" ]]; then
    echo "Aborted. Please run: pip install onnx==1.15.0"
    exit 1
  fi
fi

echo "Select export option:"
echo "1) Export ONNX (FP32)"
echo "2) Export ONNX FP16"
echo "3) Export ONNX INT8"
read -r -p "Choice [1-3]: " choice

submit_script=""
case "$choice" in
  1)
    python export_onnx.py
    submit_script="submit_with_onnx.py"
    ;;
  2)
    python export_onnx_fp16.py
    submit_script="submit_with_onnx_fp16.py"
    ;;
  3)
    if [[ ! -f "ultimate_tiled_multitask.onnx" ]]; then
      echo "FP32 ONNX not found, exporting first..."
      python export_onnx.py
    fi
    python quantize_onnx_int8.py --disable-shape-infer
    submit_script="submit_with_onnx_int8.py"
    ;;
  *)
    echo "Invalid choice: $choice"
    exit 1
    ;;
esac

read -r -p "Run test submission on the test set? [y/N]: " run_test
if [[ "$run_test" == "y" || "$run_test" == "Y" ]]; then
  python "$submit_script"
  echo "Test submission completed."
fi

echo "Done."
