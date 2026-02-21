import os
import numpy as np
import pandas as pd
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm


class Config:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, "Data")
    IMAGE_DIR = os.path.join(DATA_DIR, "images", "images")
    TEST_CSV = os.path.join(DATA_DIR, "Test.csv")
    ONNX_MODEL = os.path.join(SCRIPT_DIR, "ultimate_tiled_multitask.onnx")

    TILE_SIZE = 512
    GRID_COLS = 8
    GRID_ROWS = 6
    N_TILES = GRID_COLS * GRID_ROWS

    COUNT_COLS = [
        "Count",
        "Broken_Count",
        "Long_Count",
        "Medium_Count",
        "Black_Count",
        "Chalky_Count",
        "Red_Count",
        "Yellow_Count",
        "Green_Count",
    ]
    MEASURE_COLS = [
        "WK_Length_Average",
        "WK_Width_Average",
        "WK_LW_Ratio_Average",
        "Average_L",
        "Average_a",
        "Average_b",
    ]


def get_tiles(image: np.ndarray) -> list:
    """Split an image into a fixed 8x6 grid of tiles."""
    h, w, _ = image.shape
    step_h = h // Config.GRID_ROWS
    step_w = w // Config.GRID_COLS
    tiles = []
    for r in range(Config.GRID_ROWS):
        for c_idx in range(Config.GRID_COLS):
            y1 = r * step_h
            x1 = c_idx * step_w
            y2 = (r + 1) * step_h if r < Config.GRID_ROWS - 1 else h
            x2 = (c_idx + 1) * step_w if c_idx < Config.GRID_COLS - 1 else w
            tiles.append(image[y1:y2, x1:x2])
    return tiles


def resize_tile(tile: np.ndarray) -> np.ndarray:
    """Resize a tile to the fixed input size and return CHW float32."""
    pil_img = Image.fromarray(tile)
    pil_img = pil_img.resize((Config.TILE_SIZE, Config.TILE_SIZE), resample=Image.BILINEAR)
    arr = np.array(pil_img, dtype=np.float32)
    arr = np.transpose(arr, (2, 0, 1))
    return arr


def build_meta(comment: str) -> np.ndarray:
    """Build one-hot rice type meta vector from the Comment field."""
    rice_type = {"Paddy": 0, "White": 1, "Brown": 2}.get(comment, 0)
    meta = np.zeros((1, 3), dtype=np.float32)
    meta[0, rice_type] = 1.0
    return meta


def create_session() -> ort.InferenceSession:
    """Create ONNX Runtime session with best available providers."""
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    available = ort.get_available_providers()
    providers = [p for p in providers if p in available]
    if not providers:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(Config.ONNX_MODEL, providers=providers)


def main() -> int:
    if not os.path.exists(Config.ONNX_MODEL):
        print(f"ONNX model not found: {Config.ONNX_MODEL}")
        return 1

    test_df = pd.read_csv(Config.TEST_CSV)
    session = create_session()

    results = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        img_path = os.path.join(Config.IMAGE_DIR, f"{row['ID']}.png")
        image = np.array(Image.open(img_path).convert("RGB"))
        tiles = get_tiles(image)

        tile_tensors = np.stack([resize_tile(t) for t in tiles], axis=0)
        tile_tensors = np.expand_dims(tile_tensors, axis=0).astype(np.float32)

        meta = build_meta(row.get("Comment", ""))

        outputs = session.run(None, {"tiles": tile_tensors, "meta": meta})
        counts, measures = outputs[0], outputs[1]

        counts = counts[0]
        measures = measures[0]

        res_row = {"ID": row["ID"]}
        for i, col in enumerate(Config.COUNT_COLS):
            res_row[col] = max(0, int(round(float(counts[i]))))
        for i, col in enumerate(Config.MEASURE_COLS):
            res_row[col] = float(measures[i])

        results.append(res_row)

    out_df = pd.DataFrame(results)
    cols = ["ID"] + Config.COUNT_COLS + Config.MEASURE_COLS
    out_df = out_df[cols]
    out_df.to_csv("submission_onnx.csv", index=False)
    print("Submission saved to submission_onnx.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
