import argparse
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
    ONNX_MODEL = os.path.join(SCRIPT_DIR, "ultimate_tiled_multitask_fp16.onnx")

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


def build_meta(comment: str, dtype) -> np.ndarray:
    rice_type = {"Paddy": 0, "White": 1, "Brown": 2}.get(comment, 0)
    meta = np.zeros((1, 3), dtype=dtype)
    meta[0, rice_type] = 1.0
    return meta


def create_session(model_path: str) -> ort.InferenceSession:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    available = ort.get_available_providers()
    providers = [p for p in providers if p in available]
    if not providers:
        providers = ["CPUExecutionProvider"]
    print(f"Using providers: {providers}")
    return ort.InferenceSession(model_path, providers=providers)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ONNX inference on rice images")
    parser.add_argument("--model", default=Config.ONNX_MODEL, help="Path to ONNX model")
    parser.add_argument("--output", default="submission_onnx.csv", help="Output CSV file")
    parser.add_argument("--test-csv", default=Config.TEST_CSV, help="Test CSV file")
    parser.add_argument("--image-dir", default=Config.IMAGE_DIR, help="Image directory")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images (0=all)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ONNX model not found: {args.model}")
        return 1

    print(f"Using model: {args.model}")

    test_df = pd.read_csv(args.test_csv)
    if args.limit > 0:
        print(f"Limiting to first {args.limit} images")
        test_df = test_df.head(args.limit)

    session = create_session(args.model)

    inputs = session.get_inputs()

    tiles_input = None
    meta_input = None

    for inp in inputs:
        if hasattr(inp, "shape") and len(inp.shape) == 5:
            tiles_input = inp
        elif hasattr(inp, "shape") and len(inp.shape) == 2:
            meta_input = inp

    tiles_name = tiles_input.name if tiles_input is not None else "tiles"
    meta_name = meta_input.name if meta_input is not None else "meta"

    # ---------------------------------------------------------
    # Detect input dtype from the ONNX model
    # ---------------------------------------------------------
    # inp.type is e.g. "tensor(float16)" or "tensor(float)"
    input_type_str = tiles_input.type if tiles_input is not None else ""
    if "float16" in input_type_str:
        input_dtype = np.float16
        print("Model expects FP16 inputs")
    else:
        input_dtype = np.float32
        print("Model expects FP32 inputs")

    # Determine tile size
    expected_tile_size = Config.TILE_SIZE
    if tiles_input is not None:
        _, _, _, h, w = tiles_input.shape
        if isinstance(h, int) and isinstance(w, int):
            expected_tile_size = int(h)

    results = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):

        img_path = os.path.join(args.image_dir, f"{row['ID']}.png")
        image = np.array(Image.open(img_path).convert("RGB"))
        tiles = get_tiles(image)

        def _resize_to_expected(tile):
            pil = Image.fromarray(tile)
            pil = pil.resize(
                (expected_tile_size, expected_tile_size),
                resample=Image.BILINEAR,
            )
            arr = np.array(pil, dtype=np.float32)
            return np.transpose(arr, (2, 0, 1))

        tile_tensors = np.stack(
            [_resize_to_expected(t) for t in tiles], axis=0
        )
        tile_tensors = np.expand_dims(tile_tensors, axis=0).astype(input_dtype)

        meta = build_meta(row.get("Comment", ""), input_dtype)

        outputs = session.run(
            None,
            {
                tiles_name: tile_tensors,
                meta_name: meta,
            },
        )

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
    out_df.to_csv(args.output, index=False)

    print(f"Submission saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())