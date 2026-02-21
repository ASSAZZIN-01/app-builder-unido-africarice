import argparse
import os
import shutil

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


def disable_shape_infer() -> None:
    """Bypass ONNX shape inference to avoid model shape conflicts."""
    def _no_infer(model_path, output_path, *args, **kwargs):
        shutil.copyfile(model_path, output_path)

    onnx.shape_inference.infer_shapes_path = _no_infer


def main() -> int:
    parser = argparse.ArgumentParser(description="Dynamic INT8 quantization for ONNX model.")
    parser.add_argument(
        "--input",
        default="ultimate_tiled_multitask.onnx",
        help="Path to the FP32 ONNX model",
    )
    parser.add_argument(
        "--output",
        default="ultimate_tiled_multitask_int8.onnx",
        help="Path for the INT8 quantized ONNX model",
    )
    parser.add_argument(
        "--disable-shape-infer",
        action="store_true",
        help="Skip ONNX shape inference to avoid shape conflicts",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input model not found: {args.input}")
        return 1

    if args.disable_shape_infer:
        disable_shape_infer()

    quantize_dynamic(
        model_input=args.input,
        model_output=args.output,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
    )

    print(f"INT8 quantized model saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
