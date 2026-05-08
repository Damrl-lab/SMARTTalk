import argparse
import os
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

MODEL_FOLDER_NAME = "MB2"
DATASET_BY_MODEL_ROOT = Path("data/raw/dataset_by_model")
MODEL_ROOT = DATASET_BY_MODEL_ROOT / MODEL_FOLDER_NAME

# Failure-tag CSV (original from ssd_smart_logs/)
FAILURE_TAG_PATH = Path("data/raw/ssd_failure_tag.csv")

# Column names in CSVs
DISK_ID_COL = "disk_id"
DATE_COL = "ds"
MODEL_COL = "model"

# Model code in failure_tag file for this dataset
FAIL_MODEL_VALUE = "B2"   # in ssd_failure_tag.csv, model is A1/A2/B1/B2/C1/C2...

# SMART features for MB1/MB2 (after dropping high-NaN ones)
MODEL_FEATURES = [
    "r_5", "r_9", "r_12",
    "r_177", "r_180", "r_181", "r_182",
    "r_183", "r_184", "r_187",
    "r_195", "r_197", "r_199",
    "r_241", "r_242",
]

# Observation window size in days
WINDOW_SIZE = 30

# Horizon for "failed soon" label (days)
FAIL_HORIZON_DAYS = 30

# Split round (1, 2, or 3) from the MB1/MB2 table
# Round 1: Train months 1–22, Val 23, Test 24
# Round 2: Train months 1–21, Val 22, Test 23
# Round 3: Train months 1–20, Val 21, Test 22
SPLIT_ROUND = 3

# Files to ignore completely (all NaNs)
SKIP_FILES = {"20180101.csv", "20180102.csv"}

# Output directory
OUTPUT_DIR = Path("data/processed") / f"{MODEL_FOLDER_NAME}_round{SPLIT_ROUND}"


# ---------------------------------------------------------
# SPLIT HELPERS
# ---------------------------------------------------------

def build_month_index_map(dates: pd.Series) -> Dict[pd.Period, int]:
    """
    Build a mapping from year-month Period -> month index (1..N) in chronological order.

    Example for 2018-01 .. 2019-12:
        2018-01 -> 1
        ...
        2018-12 -> 12
        2019-01 -> 13
        ...
        2019-12 -> 24
    """
    months = sorted(dates.dt.to_period("M").unique())
    month_index_map = {m: i + 1 for i, m in enumerate(months)}

    print("Month index map (Period -> idx):")
    for m, idx in month_index_map.items():
        print(f"  {m} -> {idx}")
    return month_index_map


def assign_split_by_month(month_idx: int, round_id: int) -> Optional[str]:
    """
    Assign train/val/test using *month index* and the MB1/MB2 table:

        Round 1: Train 1–22, Val 23, Test 24
        Round 2: Train 1–21, Val 22, Test 23
        Round 3: Train 1–20, Val 21, Test 22
    """
    if month_idx is None:
        return None

    if round_id == 1:
        if 1 <= month_idx <= 22:
            return "train"
        elif month_idx == 23:
            return "val"
        elif month_idx == 24:
            return "test"
    elif round_id == 2:
        if 1 <= month_idx <= 21:
            return "train"
        elif month_idx == 22:
            return "val"
        elif month_idx == 23:
            return "test"
    elif round_id == 3:
        if 1 <= month_idx <= 20:
            return "train"
        elif month_idx == 21:
            return "val"
        elif month_idx == 22:
            return "test"

    return None


# ---------------------------------------------------------
# LOADING + NaN HANDLING
# ---------------------------------------------------------

def load_model_daily() -> pd.DataFrame:
    """
    Load all daily CSVs for one filtered model folder and concatenate them.

    Assumes each file has columns:
        disk_id, ds, model, <SMART features>

    ds is stored as an integer 20180101 → we convert via astype(str) first.
    Skips:
        20180101.csv, 20180102.csv  (all-NaN SMART attributes)

    We do not filter by model here, because this folder is already per-model.
    """
    csv_files = sorted(
        f for f in MODEL_ROOT.glob("*.csv")
        if f.is_file() and f.name not in SKIP_FILES
    )
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {MODEL_ROOT} (after skipping {SKIP_FILES})"
        )

    dfs = []
    for f in tqdm(csv_files, desc=f"Loading {MODEL_FOLDER_NAME} daily CSVs"):
        df = pd.read_csv(f)

        # schema check
        for col in [DISK_ID_COL, DATE_COL, MODEL_COL]:
            if col not in df.columns:
                raise ValueError(f"{f} missing required column {col}")

        smart_cols_present = [c for c in MODEL_FEATURES if c in df.columns]
        keep_cols = [DISK_ID_COL, DATE_COL, MODEL_COL] + smart_cols_present
        dfs.append(df[keep_cols])

    all_df = pd.concat(dfs, ignore_index=True)

    # IMPORTANT: ds is integer like 20180101 → parse via string
    all_df[DATE_COL] = pd.to_datetime(all_df[DATE_COL].astype(str), errors="coerce")

    all_df = all_df.sort_values([DISK_ID_COL, DATE_COL]).reset_index(drop=True)

    print(f"Loaded {MODEL_FOLDER_NAME} daily logs: {all_df.shape[0]} rows, "
          f"{all_df[DISK_ID_COL].nunique()} disks")
    print(f"Date range: {all_df[DATE_COL].min()}  ->  {all_df[DATE_COL].max()}")
    print(f"Models present in {MODEL_FOLDER_NAME} folder:", all_df[MODEL_COL].unique())

    # --- NaN handling for SMART features ---
    feature_cols = [c for c in MODEL_FEATURES if c in all_df.columns]

    print("\nNaN rate per feature BEFORE imputation:")
    nan_before = all_df[feature_cols].isna().mean()
    for col in feature_cols:
        print(f"  {col}: {nan_before[col]:.4f}")

    # 1) Per-disk forward-fill + backward-fill
    all_df[feature_cols] = (
        all_df.groupby(DISK_ID_COL)[feature_cols]
              .transform(lambda g: g.ffill().bfill())
    )

    # 2) Global median fallback (for columns/disks still NaN)
    feature_medians = all_df[feature_cols].median(skipna=True).fillna(0.0)
    all_df[feature_cols] = all_df[feature_cols].fillna(feature_medians)

    print("\nNaN rate per feature AFTER imputation:")
    nan_after = all_df[feature_cols].isna().mean()
    for col in feature_cols:
        print(f"  {col}: {nan_after[col]:.4f}")

    return all_df


def load_failure_tags_for_model() -> pd.DataFrame:
    """
    Load failure tags for the selected model code and keep the earliest failure per disk.
    """
    df_fail = pd.read_csv(FAILURE_TAG_PATH)

    # filter to this model code in the original tag file
    df_fail = df_fail[df_fail[MODEL_COL] == FAIL_MODEL_VALUE].copy()
    if df_fail.empty:
        print(f"WARNING: No {FAIL_MODEL_VALUE} entries in ssd_failure_tag.csv")

    # failure_time might be "2018-01-01 00:00:00" or integer 20180101
    df_fail["failure_time"] = pd.to_datetime(df_fail["failure_time"].astype(str),
                                             errors="coerce")

    # earliest failure per disk_id
    df_fail = (
        df_fail.sort_values("failure_time")
               .groupby(DISK_ID_COL, as_index=False)
               .first()[[DISK_ID_COL, "failure_time"]]
    )

    print(f"Loaded failure tags for {MODEL_FOLDER_NAME}: {df_fail.shape[0]} failed disks")
    return df_fail


def build_failure_map(df_fail: pd.DataFrame) -> Dict[int, pd.Timestamp]:
    """Map disk_id -> failure_time."""
    return dict(zip(df_fail[DISK_ID_COL].values, df_fail["failure_time"].values))


def compute_ttf_days(
    snapshot_date,
    failure_time,
) -> Optional[int]:
    """
    Compute time-to-failure in days from snapshot_date to failure_time.

    Both inputs might be numpy.datetime64 or pandas.Timestamp.
    If failure_time is None/NaT (disk never fails), returns None.
    """
    if failure_time is None or pd.isna(failure_time):
        return None

    snapshot_date = pd.Timestamp(snapshot_date).normalize()
    failure_time = pd.Timestamp(failure_time).normalize()
    delta = failure_time - snapshot_date
    return delta.days


# ---------------------------------------------------------
# WINDOWING LOGIC
# ---------------------------------------------------------

def build_windows_for_model(
    df_model: pd.DataFrame,
    failure_map: Dict[int, pd.Timestamp],
    split_round: int,
):
    """
    Build fixed-length windows (per disk) with TTF and binary labels, then split
    into train / val / test according to *month index* and split_round.
    """
    feature_cols = [c for c in MODEL_FEATURES if c in df_model.columns]
    print(f"Using SMART features for {MODEL_FOLDER_NAME}: {feature_cols}")

    # Month index map from the actual data
    month_index_map = build_month_index_map(df_model[DATE_COL])

    X_train, y_train, ttf_train = [], [], []
    X_val,   y_val,   ttf_val   = [], [], []
    X_test,  y_test,  ttf_test  = [], [], []

    total_windows = 0

    # group by disk
    for disk_id, df_disk in tqdm(
        df_model.groupby(DISK_ID_COL),
        desc=f"Building {WINDOW_SIZE}-day windows per disk",
    ):
        df_disk = df_disk.sort_values(DATE_COL).reset_index(drop=True)
        failure_time = failure_map.get(disk_id, None)

        if len(df_disk) < WINDOW_SIZE:
            continue

        dates = df_disk[DATE_COL].values
        feats = df_disk[feature_cols].values  # [T, F]

        for end_idx in range(WINDOW_SIZE - 1, len(df_disk)):
            start_idx = end_idx - WINDOW_SIZE + 1
            window_feats = feats[start_idx:end_idx + 1]  # [WINDOW_SIZE, F]
            last_date = pd.Timestamp(dates[end_idx])

            total_windows += 1

            # TTF wrt last day of window
            ttf_days = compute_ttf_days(last_date, failure_time)

            # skip windows strictly after failure
            if ttf_days is not None and ttf_days < 0:
                continue

            # binary label: 1 if failure within horizon, else 0
            if ttf_days is not None and 0 <= ttf_days <= FAIL_HORIZON_DAYS:
                label = 1
            else:
                label = 0

            # split by month index (1..24)
            month_period = last_date.to_period("M")
            month_idx = month_index_map.get(month_period, None)
            split = assign_split_by_month(month_idx, split_round)
            if split is None:
                continue

            ttf_store = ttf_days if ttf_days is not None else -1

            if split == "train":
                X_train.append(window_feats)
                y_train.append(label)
                ttf_train.append(ttf_store)
            elif split == "val":
                X_val.append(window_feats)
                y_val.append(label)
                ttf_val.append(ttf_store)
            elif split == "test":
                X_test.append(window_feats)
                y_test.append(label)
                ttf_test.append(ttf_store)

    # to numpy
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    ttf_train = np.array(ttf_train, dtype=np.int32)

    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.int64)
    ttf_val = np.array(ttf_val, dtype=np.int32)

    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int64)
    ttf_test = np.array(ttf_test, dtype=np.int32)

    print(f"\nTotal candidate windows (before split/filter): {total_windows}")
    print(f"Window counts for {MODEL_FOLDER_NAME} (month-based split):")
    print(f"  train: {X_train.shape[0]} windows")
    print(f"  val:   {X_val.shape[0]} windows")
    print(f"  test:  {X_test.shape[0]} windows")

    return (
        X_train, y_train, ttf_train,
        X_val,   y_val,   ttf_val,
        X_test,  y_test,  ttf_test,
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    global MODEL_FOLDER_NAME
    global DATASET_BY_MODEL_ROOT
    global MODEL_ROOT
    global FAILURE_TAG_PATH
    global FAIL_MODEL_VALUE
    global WINDOW_SIZE
    global FAIL_HORIZON_DAYS
    global SPLIT_ROUND
    global OUTPUT_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-folder-name",
        type=str,
        default=MODEL_FOLDER_NAME,
        help="Folder under dataset_by_model containing daily per-model CSVs, e.g. MB1 or MB2.",
    )
    parser.add_argument(
        "--dataset-by-model-root",
        type=str,
        default=str(DATASET_BY_MODEL_ROOT),
        help="Root directory containing per-model folders.",
    )
    parser.add_argument(
        "--failure-tag-path",
        type=str,
        default=str(FAILURE_TAG_PATH),
        help="Path to ssd_failure_tag.csv.",
    )
    parser.add_argument(
        "--failure-model-value",
        type=str,
        default=FAIL_MODEL_VALUE,
        help="Model code value in ssd_failure_tag.csv, e.g. B1 or B2.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=WINDOW_SIZE,
        help="Observation window size in days.",
    )
    parser.add_argument(
        "--fail-horizon-days",
        type=int,
        default=FAIL_HORIZON_DAYS,
        help="Days-to-failure horizon used for the positive label.",
    )
    parser.add_argument(
        "--split-round",
        type=int,
        default=SPLIT_ROUND,
        choices=[1, 2, 3],
        help="Temporal split round.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional explicit output directory for train/val/test .npz files.",
    )
    args = parser.parse_args()

    MODEL_FOLDER_NAME = args.model_folder_name.strip()
    DATASET_BY_MODEL_ROOT = Path(args.dataset_by_model_root)
    MODEL_ROOT = DATASET_BY_MODEL_ROOT / MODEL_FOLDER_NAME
    FAILURE_TAG_PATH = Path(args.failure_tag_path)
    FAIL_MODEL_VALUE = args.failure_model_value.strip()
    WINDOW_SIZE = int(args.window_size)
    FAIL_HORIZON_DAYS = int(args.fail_horizon_days)
    SPLIT_ROUND = int(args.split_round)
    if WINDOW_SIZE <= 0:
        raise ValueError("window-size must be a positive integer.")
    if FAIL_HORIZON_DAYS < 0:
        raise ValueError("fail-horizon-days must be non-negative.")
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    else:
        OUTPUT_DIR = Path("data/processed") / f"{MODEL_FOLDER_NAME}_round{SPLIT_ROUND}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_FOLDER_NAME} daily SMART logs from {MODEL_ROOT} ...")
    df_model = load_model_daily()

    print(f"\nLoading failure tags for model code {FAIL_MODEL_VALUE} from {FAILURE_TAG_PATH} ...")
    df_fail_model = load_failure_tags_for_model()
    failure_map = build_failure_map(df_fail_model)

    print(
        f"\nBuilding {WINDOW_SIZE}-day windows + TTF + labels + train/val/test splits "
        f"(round={SPLIT_ROUND}, fail_horizon={FAIL_HORIZON_DAYS})..."
    )
    (
        X_train, y_train, ttf_train,
        X_val,   y_val,   ttf_val,
        X_test,  y_test,  ttf_test,
    ) = build_windows_for_model(df_model, failure_map, SPLIT_ROUND)

    print("\nSaving to .npz files...")
    np.savez_compressed(
        OUTPUT_DIR / "train.npz",
        X=X_train,
        y=y_train,
        ttf=ttf_train,
        features=np.array(MODEL_FEATURES),
    )
    np.savez_compressed(
        OUTPUT_DIR / "val.npz",
        X=X_val,
        y=y_val,
        ttf=ttf_val,
        features=np.array(MODEL_FEATURES),
    )
    np.savez_compressed(
        OUTPUT_DIR / "test.npz",
        X=X_test,
        y=y_test,
        ttf=ttf_test,
        features=np.array(MODEL_FEATURES),
    )
    print(f"Saved processed {MODEL_FOLDER_NAME} windows under: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
