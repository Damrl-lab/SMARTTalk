import argparse
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm  # optional; pip install tqdm


# ---- CONFIG ----

# Root folder that contains smartlog2018ssd/ and smartlog2019ssd/
DATASET_ROOT = Path("data/raw/source_logs")

YEAR_DIRS = [
    "smartlog2018ssd",
    "smartlog2019ssd",
]

# Where to put the per-model CSVs
OUTPUT_ROOT = Path("data/raw/dataset_by_model")

# Column names in the raw CSVs
DISK_ID_COL = "disk_id"
DATE_COL = "ds"
MODEL_COL = "model"

# We want to keep only these ID/meta columns + the model-specific SMART features
ID_COLS = [DISK_ID_COL, DATE_COL, MODEL_COL]

# Base SMART attributes (for reference)
BASE_SMART_FEATURES = [
    "r_5", "r_9", "r_12", "r_173", "r_174", "r_175", "r_177", "r_180", "r_181",
    "r_182", "r_183", "r_184", "r_187", "r_195", "r_197", "r_199", "r_233",
    "r_241", "r_242",
]

# Drop lists from your NaN analysis, converted to "keep lists"

MODEL_FEATURES = {
    # A1 => drop r_177, r_181, r_182, r_183, r_233, r_241, r_242
    "MA1": [
        "r_5", "r_9", "r_12", "r_173", "r_174", "r_175",
        "r_180", "r_184", "r_187", "r_195", "r_197", "r_199",
    ],
    # A2 => drop r_173, r_177, r_180, r_181, r_182, r_195
    "MA2": [
        "r_5", "r_9", "r_12",
        "r_174", "r_175",
        "r_183", "r_184", "r_187",
        "r_197", "r_199",
        "r_233", "r_241", "r_242",
    ],
    # B1 => drop r_173, r_174, r_175, r_233
    "MB1": [
        "r_5", "r_9", "r_12",
        "r_177", "r_180", "r_181", "r_182",
        "r_183", "r_184", "r_187",
        "r_195", "r_197", "r_199",
        "r_241", "r_242",
    ],
    # B2 => drop r_173, r_174, r_175, r_233
    "MB2": [
        "r_5", "r_9", "r_12",
        "r_177", "r_180", "r_181", "r_182",
        "r_183", "r_184", "r_187",
        "r_195", "r_197", "r_199",
        "r_241", "r_242",
    ],
    # C1 => drop r_175, r_177, r_181, r_182, r_233, r_241, r_242
    "MC1": [
        "r_5", "r_9", "r_12",
        "r_173", "r_174",
        "r_180", "r_183", "r_184", "r_187",
        "r_195", "r_197", "r_199",
    ],
    # C2 => drop r_175, r_177, r_181, r_182, r_233, r_241, r_242
    "MC2": [
        "r_5", "r_9", "r_12",
        "r_173", "r_174",
        "r_180", "r_183", "r_184", "r_187",
        "r_195", "r_197", "r_199",
    ],
}


def ensure_output_dirs():
    """Create one folder per model under OUTPUT_ROOT."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for model in MODEL_FEATURES.keys():
        (OUTPUT_ROOT / model).mkdir(exist_ok=True)


def iter_daily_files():
    """Yield (year_dir, csv_path) for all daily CSVs."""
    for year_dir in YEAR_DIRS:
        year_path = DATASET_ROOT / year_dir
        if not year_path.is_dir():
            raise FileNotFoundError(f"Expected folder not found: {year_path}")
        for fname in sorted(os.listdir(year_path)):
            if not fname.endswith(".csv"):
                continue
            yield year_dir, year_path / fname


def process_all_days():
    ensure_output_dirs()

    rows_per_model = {m: 0 for m in MODEL_FEATURES.keys()}

    daily_files = list(iter_daily_files())
    for year_dir, csv_path in tqdm(daily_files, desc="Processing daily SMART logs"):
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] Failed to read {csv_path}: {e}")
            continue

        # Schema check for the ID/meta columns
        missing_meta = [c for c in ID_COLS if c not in df.columns]
        if missing_meta:
            raise ValueError(
                f"{csv_path} is missing required columns {missing_meta}. "
                f"Adjust DISK_ID_COL/DATE_COL/MODEL_COL in the script."
            )

        # For each model: filter rows and keep only requested SMART features
        for model, feature_list in MODEL_FEATURES.items():
            sub = df[df[MODEL_COL] == model]
            if sub.empty:
                continue

            # Keep only the SMART columns you specified (if they exist in this file)
            smart_cols = [c for c in feature_list if c in sub.columns]
            keep_cols = ID_COLS + smart_cols
            sub = sub[keep_cols]

            out_path = OUTPUT_ROOT / model / csv_path.name
            sub.to_csv(out_path, index=False)

            rows_per_model[model] += len(sub)

    print("\nDone. Rows per model (over all days):")
    for model, nrows in rows_per_model.items():
        print(f"  {model}: {nrows:,} rows")


def main() -> None:
    global DATASET_ROOT
    global OUTPUT_ROOT
    global YEAR_DIRS

    parser = argparse.ArgumentParser(
        description="Split raw Alibaba SMART daily logs into per-model folders used by SMARTTalk.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(DATASET_ROOT),
        help="Folder containing smartlog2018ssd/ and smartlog2019ssd/.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(OUTPUT_ROOT),
        help="Folder where per-model daily CSVs are written.",
    )
    parser.add_argument(
        "--year-dirs",
        nargs="+",
        default=YEAR_DIRS,
        help="Subdirectories under dataset-root that contain daily SMART CSVs.",
    )
    args = parser.parse_args()

    DATASET_ROOT = Path(args.dataset_root)
    OUTPUT_ROOT = Path(args.output_root)
    YEAR_DIRS = list(args.year_dirs)

    process_all_days()


if __name__ == "__main__":
    main()
