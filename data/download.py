"""
Downloads NASA C-MAPSS FD001 dataset.
Source: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
Manual alternative: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
"""

import os
import urllib.request
import zipfile

DATA_DIR = os.path.join(os.path.dirname(__file__), "raw")

# Direct mirror via Kaggle API or manual download
KAGGLE_DATASET = "behrad3d/nasa-cmaps"

EXPECTED_FILES = [
    "train_FD001.txt",
    "test_FD001.txt",
    "RUL_FD001.txt",
]


def already_downloaded():
    return all(os.path.exists(os.path.join(DATA_DIR, f)) for f in EXPECTED_FILES)


def download_via_kaggle():
    try:
        import kaggle
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"Downloading {KAGGLE_DATASET} ...")
        kaggle.api.dataset_download_files(KAGGLE_DATASET, path=DATA_DIR, unzip=True)
        print("Done.")
    except ImportError:
        print("kaggle package not installed. Run: pip install kaggle")
        print("Then place your kaggle.json API key in ~/.kaggle/kaggle.json")
        raise
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        print_manual_instructions()
        raise


def print_manual_instructions():
    print("\n--- Manual download ---")
    print("1. Go to https://www.kaggle.com/datasets/behrad3d/nasa-cmaps")
    print("2. Download and extract the zip")
    print(f"3. Place train_FD001.txt, test_FD001.txt, RUL_FD001.txt in: {DATA_DIR}")
    print("-----------------------\n")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    if already_downloaded():
        print("Files already present, skipping download.")
    else:
        try:
            download_via_kaggle()
        except Exception:
            print_manual_instructions()
