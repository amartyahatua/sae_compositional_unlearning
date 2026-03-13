"""
TOFU Dataset Utility
====================
Downloads the TOFU dataset from HuggingFace, saves each split as a CSV,
and provides a function to reload them as a HuggingFace DatasetDict.

TOFU (Task of Fictitious Unlearning) — locuslab/TOFU
Configs available: full, forget01, forget05, forget10,
                   retain90, retain95, retain99, world_facts

Usage:
    # Step 1: Download and save CSVs
    download_tofu_as_csv(config="full", output_dir="tofu_data")

    # Step 2: Load CSVs back as HuggingFace DatasetDict
    dataset = load_tofu_from_csv(data_dir="tofu_data", config="full")
    print(dataset)
    print(dataset["train"][0])
"""

import os
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict


# ── 1. DOWNLOAD AND SAVE AS CSV ──────────────────────────────────────────────

def download_tofu_as_csv(
    config: str = "full",
    output_dir: str = "tofu_data",
    splits: list = None
) -> dict:
    """
    Downloads the TOFU dataset from HuggingFace and saves each split as a CSV.

    Args:
        config     : TOFU config name. One of:
                     'full', 'forget01', 'forget05', 'forget10',
                     'retain90', 'retain95', 'retain99', 'world_facts'
        output_dir : Directory to save CSV files into.
        splits     : List of splits to download. If None, downloads all
                     available splits for the given config.

    Returns:
        dict mapping split name -> path to saved CSV file.

    Example:
        paths = download_tofu_as_csv(config="full", output_dir="tofu_data")
        # Saves:
        #   tofu_data/full_train.csv
        #   tofu_data/full_test.csv
        #   (TOFU 'full' only has train; forget/retain configs vary)
    """
    print(f"Loading TOFU config='{config}' from HuggingFace...")
    ds = load_dataset("locuslab/TOFU", config)
    print(f"Splits found: {list(ds.keys())}")

    os.makedirs(output_dir, exist_ok=True)

    saved_paths = {}
    target_splits = splits if splits else list(ds.keys())

    for split in target_splits:
        if split not in ds:
            print(f"  Warning: split '{split}' not found in config '{config}', skipping.")
            continue

        filename = f"{config}_{split}.csv"
        filepath = os.path.join(output_dir, filename)

        df = ds[split].to_pandas()
        df.to_csv(filepath, index=False)

        saved_paths[split] = filepath
        print(f"  Saved {split}: {len(df)} rows → {filepath}")

    print(f"\nAll CSVs saved to: {os.path.abspath(output_dir)}/")
    return saved_paths


# ── 2. LOAD CSVs BACK AS HUGGINGFACE DATASETDICT ─────────────────────────────

def load_tofu_from_csv(
    data_dir: str = "tofu_data",
    config: str = "full",
    splits: list = None
) -> DatasetDict:
    """
    Reads TOFU CSV files saved by download_tofu_as_csv() and returns a
    HuggingFace DatasetDict — identical in structure to load_dataset().

    Args:
        data_dir : Directory containing the CSV files.
        config   : TOFU config name used when saving (e.g. 'full').
        splits   : Splits to load. If None, auto-detects from files in data_dir.

    Returns:
        DatasetDict with one Dataset per split.

    Example:
        dataset = load_tofu_from_csv(data_dir="tofu_data", config="full")
        print(dataset)
        # DatasetDict({
        #     train: Dataset({features: ['question', 'answer'], num_rows: 4000})
        # })
        print(dataset["train"][0])
        # {'question': '...', 'answer': '...'}
    """
    # Auto-detect splits if not specified
    if splits is None:
        splits = _detect_splits(data_dir, config)
        if not splits:
            raise FileNotFoundError(
                f"No CSV files found for config='{config}' in '{data_dir}'.\n"
                f"Run download_tofu_as_csv(config='{config}', output_dir='{data_dir}') first."
            )

    dataset_dict = {}

    for split in splits:
        filename = f"{config}_{split}.csv"
        filepath = os.path.join(data_dir, filename)

        if not os.path.exists(filepath):
            print(f"  Warning: {filepath} not found, skipping split '{split}'.")
            continue

        df = pd.read_csv(filepath)
        dataset_dict[split] = Dataset.from_pandas(df, preserve_index=False)
        print(f"  Loaded {split}: {len(df)} rows from {filepath}")

    if not dataset_dict:
        raise FileNotFoundError(
            f"No valid CSV files could be loaded for config='{config}' "
            f"from '{data_dir}'."
        )

    return DatasetDict(dataset_dict)


# ── 3. CONVENIENCE: DOWNLOAD ALL CONFIGS ─────────────────────────────────────

def download_all_configs(output_dir: str = "tofu_data") -> dict:
    """
    Downloads all standard TOFU configs and saves them as CSVs.

    Configs downloaded:
        full, forget01, forget05, forget10,
        retain90, retain95, retain99, world_facts

    Returns:
        Nested dict: {config: {split: filepath}}
    """
    configs = [
        "full",
        "forget01", "forget05", "forget10",
        "retain90", "retain95", "retain99",
        "world_facts"
    ]

    all_paths = {}
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Config: {config}")
        print(f"{'='*50}")
        try:
            paths = download_tofu_as_csv(config=config, output_dir=output_dir)
            all_paths[config] = paths
        except Exception as e:
            print(f"  Failed to download config '{config}': {e}")

    return all_paths


# ── 4. HELPER ─────────────────────────────────────────────────────────────────

def _detect_splits(data_dir: str, config: str) -> list:
    """Auto-detect available splits by scanning CSV filenames."""
    if not os.path.exists(data_dir):
        return []
    prefix = f"{config}_"
    splits = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.startswith(prefix) and fname.endswith(".csv"):
            split = fname[len(prefix):-4]  # strip prefix and .csv
            splits.append(split)
    return splits


# ── 5. QUICK INSPECTION UTILITY ───────────────────────────────────────────────

def inspect_tofu(dataset: DatasetDict) -> None:
    """
    Prints a summary of a loaded TOFU DatasetDict.

    Args:
        dataset : DatasetDict returned by load_tofu_from_csv()
    """
    print("\n" + "="*50)
    print("TOFU Dataset Summary")
    print("="*50)
    print(dataset)
    print()

    for split, ds in dataset.items():
        print(f"── Split: {split} ──")
        print(f"   Rows    : {len(ds)}")
        print(f"   Columns : {list(ds.features.keys())}")
        print(f"   Example :")
        example = ds[0]
        for col, val in example.items():
            preview = str(val)[:120] + "..." if len(str(val)) > 120 else str(val)
            print(f"     {col}: {preview}")
        print()


# ── MAIN — demo run ───────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Step 1: Download TOFU 'full' config and save as CSV ──
    print("Step 1: Downloading TOFU and saving as CSV...")
    paths = download_tofu_as_csv(
        config="retain90",
        output_dir="tofu_data"
    )
    download_all_configs(output_dir="tofu_data")
    # ── Step 2: Load CSVs back as HuggingFace DatasetDict ──
    print("\nStep 2: Loading CSVs as HuggingFace DatasetDict...")
    dataset = load_tofu_from_csv(
        data_dir="tofu_data",
        config="full"
    )

    # ── Step 3: Inspect ──
    inspect_tofu(dataset)

    # ── Step 4: Use like any HuggingFace dataset ──
    print("Step 4: Accessing data...")
    train = dataset["train"]
    print(f"First example: {train[0]}")
    print(f"Columns: {train.column_names}")
    print(f"PyTorch DataLoader ready: {train.with_format('torch')}")