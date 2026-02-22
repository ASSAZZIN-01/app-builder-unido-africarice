# NOTE: This script requires onnx==1.15.0 to produce IR version 9 models,
# which are compatible with mobile ONNX Runtime. Do not use onnx >= 1.16.0.

import argparse
import os

import torch

from export_onnx import ExportWrapper
from submit import Config, UltimateSpecialist


def main() -> int:
    parser = argparse.ArgumentParser(description="Export FP16 ONNX model from PyTorch.")
    parser.add_argument(
        "--output",
        default="ultimate_tiled_multitask_fp16.onnx",
        help="Path for the FP16 ONNX model",
    )
    args = parser.parse_args()

    checkpoint_path = os.path.join(Config.SCRIPT_DIR, Config.CHECKPOINT)
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return 1

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = UltimateSpecialist(Config.MODEL_NAME)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    wrapper = ExportWrapper(model, checkpoint["m_stats"][0], checkpoint["m_stats"][1])
    wrapper.eval()

    wrapper = wrapper.half()

    dummy_tiles = torch.zeros(
        1,
        Config.N_TILES,
        3,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
        dtype=torch.float16,
    )
    dummy_meta = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float16)

    torch.onnx.export(
        wrapper,
        (dummy_tiles, dummy_meta),
        args.output,
        export_params=True,
        opset_version=18,
        do_constant_folding=False,
        input_names=["tiles", "meta"],
        output_names=["counts", "measures"],
        dynamic_axes={
            "tiles": {0: "batch"},
            "meta": {0: "batch"},
            "counts": {0: "batch"},
            "measures": {0: "batch"},
        },
        dynamo=False,
    )

    print(f"FP16 model exported to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
