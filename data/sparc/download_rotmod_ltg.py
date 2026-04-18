from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import requests


DEFAULT_URL = "https://astroweb.cwru.edu/SPARC/Rotmod_LTG.zip"


def download(url: str, out_zip: Path) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with out_zip.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def extract(zip_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract SPARC Rotmod_LTG data.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--zip", dest="zip_path", default="Rotmod_LTG.zip")
    parser.add_argument("--out", dest="out_dir", default="Rotmod_LTG")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    zip_path = root / args.zip_path
    out_dir = root / args.out_dir

    download(args.url, zip_path)
    extract(zip_path, out_dir)

    print(f"Downloaded: {zip_path}")
    print(f"Extracted to: {out_dir}")


if __name__ == "__main__":
    main()
