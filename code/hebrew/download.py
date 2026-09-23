"""
Step 1: Download HeQ, ParaShoot, and MKQA (Hebrew subset).
Run once. Saves raw data to data/raw/.
"""
import gzip
import json
import shutil
import urllib.request
from pathlib import Path

from datasets import load_dataset

RAW = Path("data/raw")

PARASHOOT_BASE = "https://raw.githubusercontent.com/omrikeren/ParaShoot/main/data"
MKQA_URL = "https://github.com/apple/ml-mkqa/raw/main/dataset/mkqa.jsonl.gz"


def download_heq():
    print("==> Downloading HeQ from HuggingFace (pig4431/HeQ_v1)...")
    ds = load_dataset("pig4431/HeQ_v1")
    out = RAW / "heq"
    for split in ds:
        path = out / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in ds[split]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"   saved {split}: {len(ds[split])} examples -> {path}")


def download_parashoot():
    print("==> Downloading ParaShoot from GitHub (omrikeren/ParaShoot)...")
    out = RAW / "parashoot"
    for split in ("train", "dev", "test"):
        url = f"{PARASHOOT_BASE}/{split}.json"
        dest = out / f"{split}.json"
        print(f"   fetching {url}")
        urllib.request.urlretrieve(url, dest)
        with open(dest, encoding="utf-8") as f:
            data = json.load(f)
        n = len(data["data"])
        print(f"   saved {split}: {n} QA pairs -> {dest}")


def download_mkqa():
    print("==> Downloading MKQA from GitHub (apple/ml-mkqa)...")
    out = RAW / "mkqa"
    gz_path = out / "mkqa.jsonl.gz"
    dest = out / "mkqa.jsonl"
    print(f"   fetching {MKQA_URL}")
    urllib.request.urlretrieve(MKQA_URL, gz_path)
    with gzip.open(gz_path, "rb") as f_in, open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()
    n = sum(1 for _ in open(dest, encoding="utf-8"))
    print(f"   saved {n} examples -> {dest}")


if __name__ == "__main__":
    download_heq()
    download_parashoot()
    download_mkqa()
    print("\nAll datasets downloaded successfully.")
