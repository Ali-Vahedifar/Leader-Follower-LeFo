#!/usr/bin/env python3
"""
Dataset Download Script for LeFo Framework.

Downloads haptic teleoperation datasets from Zenodo for training
and evaluation of Leader-Follower signal prediction models.

Author: Mohammad Ali Vahedifar (av@ece.au.dk)
Co-Author: Qi Zhang (qz@ece.au.dk)
Institution: DIGIT and Department of Electrical and Computer Engineering,
             Aarhus University, Denmark

Dataset DOI: 10.5281/zenodo.14924062

Usage:
    python download_data.py --output-dir ./data --dataset all

Copyright (c) 2025 Mohammad Ali Vahedifar
"""

import argparse
import hashlib
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

# Zenodo DOI and dataset information
ZENODO_DOI = "10.5281/zenodo.14924062"
ZENODO_RECORD_ID = "14924062"
ZENODO_BASE_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

# Dataset file mappings
DATASETS = {
    "drag": {
        "filename": "drag_max_stiffness_y.csv",
        "description": "Dragging with maximum stiffness in Y direction",
        "samples": 10000,
        "md5": None  # Will be verified from Zenodo
    },
    "horizontal_fast": {
        "filename": "horizontal_movement_fast.csv",
        "description": "Fast horizontal free-air movement",
        "samples": 8000,
        "md5": None
    },
    "horizontal_slow": {
        "filename": "horizontal_movement_slow.csv",
        "description": "Slow horizontal free-air movement",
        "samples": 12000,
        "md5": None
    },
    "tap_hold_fast": {
        "filename": "tap_and_hold_z_fast.csv",
        "description": "Fast tapping and holding in Z direction",
        "samples": 9000,
        "md5": None
    },
    "tap_hold_slow": {
        "filename": "tap_and_hold_z_slow.csv",
        "description": "Slow tapping and holding in Z direction",
        "samples": 11000,
        "md5": None
    },
    "tapping_yz": {
        "filename": "tapping_max_y_z.csv",
        "description": "Tapping with maximum stiffness in Y and Z",
        "samples": 10000,
        "md5": None
    },
    "tapping_z": {
        "filename": "tapping_max_z.csv",
        "description": "Tapping with maximum stiffness in Z direction",
        "samples": 10000,
        "md5": None
    }
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download LeFo haptic datasets from Zenodo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available datasets:
  drag            - Dragging with maximum stiffness (Y axis)
  horizontal_fast - Fast horizontal free-air movement
  horizontal_slow - Slow horizontal free-air movement
  tap_hold_fast   - Fast tapping and holding (Z axis)
  tap_hold_slow   - Slow tapping and holding (Z axis)
  tapping_yz      - Tapping with max stiffness (Y and Z axes)
  tapping_z       - Tapping with max stiffness (Z axis)
  all             - Download all datasets

Zenodo DOI: {ZENODO_DOI}
        """
    )
    
    parser.add_argument(
        "--output-dir", type=str, default="./data",
        help="Directory to save downloaded datasets"
    )
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["all"] + list(DATASETS.keys()),
        help="Which dataset to download"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify MD5 checksums after download"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-download even if files exist"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output"
    )
    
    return parser.parse_args()


def download_progress(count, block_size, total_size):
    """Display download progress."""
    percent = int(count * block_size * 100 / total_size)
    percent = min(percent, 100)
    sys.stdout.write(f"\r  Progress: {percent}%")
    sys.stdout.flush()


def compute_md5(filepath):
    """Compute MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_zenodo_file_urls():
    """Fetch file URLs from Zenodo API."""
    import json
    from urllib.request import urlopen
    
    try:
        with urlopen(ZENODO_BASE_URL) as response:
            data = json.loads(response.read().decode())
        
        files = {}
        for file_info in data.get('files', []):
            filename = file_info['key']
            files[filename] = {
                'url': file_info['links']['self'],
                'size': file_info['size'],
                'checksum': file_info.get('checksum', '').replace('md5:', '')
            }
        return files
    except Exception as e:
        print(f"Warning: Could not fetch Zenodo metadata: {e}")
        return None


def download_file(url, output_path, quiet=False):
    """Download a file from URL."""
    try:
        if quiet:
            urlretrieve(url, output_path)
        else:
            urlretrieve(url, output_path, reporthook=download_progress)
            print()  # New line after progress
        return True
    except URLError as e:
        print(f"Error downloading: {e}")
        return False


def download_dataset(dataset_key, output_dir, zenodo_files=None, verify=False, force=False, quiet=False):
    """Download a single dataset."""
    dataset_info = DATASETS[dataset_key]
    filename = dataset_info['filename']
    output_path = output_dir / filename
    
    if not quiet:
        print(f"\n{dataset_key}: {dataset_info['description']}")
        print(f"  File: {filename}")
        print(f"  Samples: ~{dataset_info['samples']:,}")
    
    # Check if file exists
    if output_path.exists() and not force:
        if not quiet:
            print(f"  Status: Already exists (use --force to re-download)")
        return True
    
    # Get download URL
    if zenodo_files and filename in zenodo_files:
        url = zenodo_files[filename]['url']
        expected_checksum = zenodo_files[filename].get('checksum')
    else:
        # Fallback to direct Zenodo URL pattern
        url = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/{filename}?download=1"
        expected_checksum = None
    
    if not quiet:
        print(f"  Downloading from Zenodo...")
    
    # Download
    success = download_file(url, output_path, quiet)
    
    if not success:
        return False
    
    # Verify checksum if requested
    if verify and expected_checksum:
        if not quiet:
            print(f"  Verifying checksum...")
        actual_checksum = compute_md5(output_path)
        if actual_checksum != expected_checksum:
            print(f"  WARNING: Checksum mismatch!")
            print(f"    Expected: {expected_checksum}")
            print(f"    Actual:   {actual_checksum}")
            return False
        if not quiet:
            print(f"  Checksum verified: OK")
    
    if not quiet:
        print(f"  Status: Downloaded successfully")
    
    return True


def create_dataset_info(output_dir):
    """Create a dataset info JSON file."""
    import json
    
    info = {
        "name": "LeFo Haptic Teleoperation Dataset",
        "version": "1.0",
        "doi": ZENODO_DOI,
        "description": "Haptic teleoperation data collected using Novint Falcon device with Chai3D",
        "citation": "Vahedifar, M.A., Zhang, Q. (2025). Leader-Follower Signal Prediction for Tactile Internet. IEEE MLSP 2025.",
        "features": {
            "position": ["x", "y", "z"],
            "velocity": ["x", "y", "z"],
            "force": ["x", "y", "z"]
        },
        "datasets": DATASETS
    }
    
    info_path = output_dir / "dataset_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    return info_path


def main():
    """Main download function."""
    args = parse_args()
    
    print("="*60)
    print("LeFo Dataset Downloader")
    print("="*60)
    print(f"Zenodo DOI: {ZENODO_DOI}")
    print(f"Output directory: {args.output_dir}")
    print("="*60)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch Zenodo metadata
    if not args.quiet:
        print("\nFetching dataset metadata from Zenodo...")
    zenodo_files = get_zenodo_file_urls()
    
    # Determine datasets to download
    if args.dataset == "all":
        datasets_to_download = list(DATASETS.keys())
    else:
        datasets_to_download = [args.dataset]
    
    # Download datasets
    success_count = 0
    fail_count = 0
    
    for dataset_key in datasets_to_download:
        success = download_dataset(
            dataset_key=dataset_key,
            output_dir=output_dir,
            zenodo_files=zenodo_files,
            verify=args.verify,
            force=args.force,
            quiet=args.quiet
        )
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # Create dataset info file
    info_path = create_dataset_info(output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Dataset info: {info_path}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    if fail_count > 0:
        print("\nSome downloads failed. Please try again or download manually from:")
        print(f"  https://zenodo.org/record/{ZENODO_RECORD_ID}")
        sys.exit(1)
    else:
        print("\nAll datasets downloaded successfully!")


if __name__ == "__main__":
    main()
