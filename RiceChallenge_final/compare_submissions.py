import argparse
import pandas as pd


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


def summarize_diff(df_a: pd.DataFrame, df_b: pd.DataFrame, cols: list) -> pd.DataFrame:
    diffs = {}
    for col in cols:
        delta = (df_a[col].astype(float) - df_b[col].astype(float)).abs()
        diffs[col] = {
            "max_abs": float(delta.max()),
            "mean_abs": float(delta.mean()),
            "median_abs": float(delta.median()),
        }
    return pd.DataFrame(diffs).T


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two submission CSVs.")
    parser.add_argument("--a", default="submission.csv", help="First submission CSV")
    parser.add_argument("--b", default="submission_onnx.csv", help="Second submission CSV")
    args = parser.parse_args()

    df_a = pd.read_csv(args.a)
    df_b = pd.read_csv(args.b)

    if "ID" not in df_a.columns or "ID" not in df_b.columns:
        print("Both files must include an ID column.")
        return 1

    df_a = df_a.sort_values("ID").reset_index(drop=True)
    df_b = df_b.sort_values("ID").reset_index(drop=True)

    if len(df_a) != len(df_b):
        print(f"Row count mismatch: {len(df_a)} vs {len(df_b)}")
        return 1

    all_cols = ["ID"] + COUNT_COLS + MEASURE_COLS
    missing_a = [c for c in all_cols if c not in df_a.columns]
    missing_b = [c for c in all_cols if c not in df_b.columns]
    if missing_a or missing_b:
        print(f"Missing columns. a: {missing_a} b: {missing_b}")
        return 1

    if not (df_a["ID"].values == df_b["ID"].values).all():
        print("ID ordering mismatch after sort.")
        return 1

    count_summary = summarize_diff(df_a, df_b, COUNT_COLS)
    measure_summary = summarize_diff(df_a, df_b, MEASURE_COLS)

    print("Count columns diff summary:")
    print(count_summary)
    print("\nMeasure columns diff summary:")
    print(measure_summary)

    total_delta = 0.0
    for col in COUNT_COLS + MEASURE_COLS:
        total_delta += (df_a[col].astype(float) - df_b[col].astype(float)).abs().sum()
    print(f"\nTotal absolute difference sum: {total_delta}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
