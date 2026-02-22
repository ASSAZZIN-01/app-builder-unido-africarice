# NOTE: This script requires onnx==1.15.0 to produce IR version 9 models,
# which are compatible with mobile ONNX Runtime. Do not use onnx >= 1.16.0.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from submit import Config, UltimateSpecialist


class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, m_mean, m_std):
        super().__init__()
        self.model = model
        self.register_buffer("m_mean", torch.tensor(m_mean, dtype=torch.float32))
        self.register_buffer("m_std", torch.tensor(m_std, dtype=torch.float32))

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
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tiles.dtype, device=tiles.device)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tiles.dtype, device=tiles.device)
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

    dummy_tiles = torch.zeros(
        1,
        Config.N_TILES,
        3,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
        dtype=torch.float32,
    )
    dummy_meta = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)

    onnx_path = os.path.join(Config.SCRIPT_DIR, "ultimate_tiled_multitask.onnx")
    torch.onnx.export(
        wrapper,
        (dummy_tiles, dummy_meta),
        onnx_path,
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

    print(f"ONNX model exported to: {onnx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
