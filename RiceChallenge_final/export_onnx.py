# NOTE: This script requires onnx==1.15.0 to produce IR version 9 models,
# which are compatible with mobile ONNX Runtime. Do not use onnx >= 1.16.0.

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import numpy as _np
    _np_v = _np.__version__
    if int(_np_v.split(".")[0]) >= 2:
        print("Detected NumPy >= 2.0 which may be incompatible with installed compiled modules (torch).")
        print("Please install a NumPy < 2.0 version (e.g. pip install numpy==1.26.4) in this environment and retry.")
        sys.exit(1)
except Exception:
    # if numpy is completely unavailable, let torch import fail later with its own message
    pass

from submit import Config, UltimateSpecialist

try:
    import onnx
    from onnx import TensorProto
    from onnx.external_data_helper import load_external_data_for_model
except Exception:
    onnx = None



class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, m_mean, m_std):
        super().__init__()
        self.model = model
        self.register_buffer("m_mean", torch.tensor(m_mean, dtype=torch.float32))
        self.register_buffer("m_std", torch.tensor(m_std, dtype=torch.float32))

        # register ImageNet mean/std as buffers to avoid creating tensors inside forward
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32))

        paddy_vec = [0.0 if c in Config.PADDY_ZERO else 1.0 for c in Config.COUNT_COLS]
        brown_vec = [0.0 if c in Config.BROWN_ZERO else 1.0 for c in Config.COUNT_COLS]
        self.register_buffer("paddy_vec", torch.tensor(paddy_vec, dtype=torch.float32))
        self.register_buffer("brown_vec", torch.tensor(brown_vec, dtype=torch.float32))
        self.register_buffer("ones_vec", torch.ones(len(Config.COUNT_COLS), dtype=torch.float32))

    def forward(self, tiles: torch.Tensor, meta: torch.Tensor):
        """
        tiles: (B, 48, 3, 512, 512) float32 in [0, 255]
        meta: (B, 3) one-hot for rice type [Paddy, White, Brown]
        """
        tiles = tiles / 255.0
        mean = self.imagenet_mean.to(tiles.dtype).to(tiles.device)
        std = self.imagenet_std.to(tiles.dtype).to(tiles.device)
        tiles = (tiles - mean.view(1, 1, 3, 1, 1)) / std.view(1, 1, 3, 1, 1)
        counts, measures = self.model(tiles, meta)

        counts = counts / Config.SCALE
        measures = measures * (self.m_std + 1e-8) + self.m_mean

        rice_type = torch.argmax(meta, dim=1)
        rice_onehot = F.one_hot(rice_type, num_classes=3).to(counts.dtype)
        mask = (
            rice_onehot[:, 0:1] * self.paddy_vec
            + rice_onehot[:, 1:2] * self.ones_vec
            + rice_onehot[:, 2:3] * self.brown_vec
        )
        counts = counts * mask
        return counts, measures


def main() -> int:
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

    # Use the same preprocessing tile size as training (`Config.TILE_SIZE`) to
    # preserve behavior exactly as in `submit.py`. To reduce memory pressure
    # during tracing, keep the dummy `n_tiles` small (1) while marking the
    # tiles axis dynamic so the exported model accepts the full tile count at
    # runtime.
    export_n_tiles = 1
    export_tile_size = Config.TILE_SIZE
    dummy_tiles = torch.zeros(
        1,
        export_n_tiles,
        3,
        export_tile_size,
        export_tile_size,
        dtype=torch.float32,
    )
    dummy_meta = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)

    onnx_path = os.path.join(Config.SCRIPT_DIR, "ultimate_tiled_multitask.onnx")
    torch.onnx.export(
        wrapper,
        (dummy_tiles, dummy_meta),
        onnx_path,
        export_params=True,
        # opset 17 is broadly supported by torch.onnx.export; use 17 for compatibility
        opset_version=17,
        do_constant_folding=False,
        input_names=["tiles", "meta"],
        output_names=["counts", "measures"],
        dynamic_axes={
            "tiles": {0: "batch", 1: "n_tiles"},
            "meta": {0: "batch"},
            "counts": {0: "batch"},
            "measures": {0: "batch"},
        },
        # some torch versions do not accept the `dynamo` kwarg; omitted for compatibility
    )
    print(f"ONNX model exported to: {onnx_path}")

    # Post-check: ensure model is embedded (single-file) for easy mobile deployment.
    if onnx is None:
        print("Note: onnx package not available in this environment; cannot verify external-data usage.")
    else:
        model = onnx.load(onnx_path)
        uses_external = any(
            (getattr(init, "data_location", None) == TensorProto.EXTERNAL) for init in model.graph.initializer
        )
        if uses_external:
            print("Model uses external data. Attempting to embed external tensors into a single .onnx file...")
            try:
                # load external binary data referenced by the file (expects .data file alongside)
                load_external_data_for_model(model, base_dir=os.path.dirname(onnx_path))
                onnx.save_model(model, onnx_path)
                print("External data embedded into single ONNX file.")
            except Exception as e:
                print("Failed to embed external data:", e)
                print("You will need to provide the external data files alongside the .onnx or resolve external references.")
        else:
            print("Model does not use external data (single-file .onnx).")

    try:
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"Final ONNX file size: {size_mb:.2f} MB")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
