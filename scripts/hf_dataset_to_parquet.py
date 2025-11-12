#!/usr/bin/env python3
"""
Convert HuggingFace dataset to sharded parquet files for streaming compatibility.

Usage:
    python hf_dataset_to_parquet.py <input_dir> <output_dir> [--num-shards N]
"""

import argparse
from pathlib import Path
from datasets import load_from_disk


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace dataset to sharded parquet files"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to HuggingFace dataset directory"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to output directory for parquet shards"
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Number of shards (default: use dataset's natural sharding)"
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.input_dir}...")
    dataset = load_from_disk(args.input_dir)

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine number of shards
    if args.num_shards is None:
        # Try to detect shards from input directory
        input_path = Path(args.input_dir)
        # Check for arrow files in both root and data/ subdirectory
        shard_files = list(input_path.glob("data-*.arrow"))
        if not shard_files:
            data_dir = input_path / "data"
            if data_dir.exists():
                shard_files = list(data_dir.glob("data-*.arrow"))

        num_shards = len(shard_files) if shard_files else 1
        if num_shards > 1:
            print(f"Auto-detected {num_shards} shards from input directory")
    else:
        num_shards = args.num_shards

    print(f"Exporting to {num_shards} parquet shard(s)...")

    if num_shards == 1:
        # Single file export
        output_file = output_path / "dataset.parquet"
        dataset.to_parquet(str(output_file))
        print(f"✓ Saved to {output_file}")
    else:
        # Sharded export
        for shard_idx in range(num_shards):
            shard = dataset.shard(num_shards=num_shards, index=shard_idx)
            output_file = output_path / f"dataset_shard_{shard_idx:04d}.parquet"
            shard.to_parquet(str(output_file))
            print(f"✓ Saved shard {shard_idx + 1}/{num_shards} to {output_file}")

    print("Done!")


if __name__ == "__main__":
    main()
