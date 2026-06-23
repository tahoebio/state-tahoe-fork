#!/usr/bin/env python
"""
fix_obs_index_inplace.py — Retrofit already-exported Caspian state_input h5ads so
cell_load discovers the real cell barcodes, WITHOUT regenerating them.

The exports were written with a named obs index ("obs_id"), so AnnData stored the
barcodes at obs/obs_id and set obs.attrs["_index"] = "obs_id". cell_load's
_load_cell_barcodes only reads the literal obs/_index key, so it falls back to
generic "cell_000000" names. This script renames that dataset in place:

    obs/obs_id            -> obs/_index          (HDF5 dataset rename, no copy)
    obs.attrs["_index"]   = "_index"

It is metadata-only and idempotent: files already exposing obs/_index are skipped.
The barcode VALUES are untouched; only the key AnnData/cell_load look under changes.

IMPORTANT: this opens each file read/write ("r+"). Do NOT run it while a training
job has these h5ads open — concurrent HDF5 writers/readers can crash or corrupt.
Run it in a quiet window (e.g. during the cluster reboot, before relaunching).

Usage:
    python fix_obs_index_inplace.py                       # fix all *.h5ad in default dir
    python fix_obs_index_inplace.py --dir /path/to/dir
    python fix_obs_index_inplace.py file1.h5ad file2.h5ad # specific files
    python fix_obs_index_inplace.py --dry-run             # report only, change nothing
"""
import argparse
import glob
import os
import sys

import h5py

DEFAULT_DIR = "/nvme-shared/datasets/caspian/state_input/"


def fix_file(path: str, dry_run: bool) -> str:
    """Returns a one-word status: skipped / fixed / would-fix / error:<reason>."""
    try:
        with h5py.File(path, "r+" if not dry_run else "r") as f:
            if "obs" not in f:
                return "error:no-obs-group"
            obs = f["obs"]

            # Already fixed (or natively unnamed index) -> nothing to do.
            if "_index" in obs:
                return "skipped"

            idx_name = obs.attrs.get("_index")
            if isinstance(idx_name, bytes):
                idx_name = idx_name.decode()
            if not idx_name:
                return "error:no-_index-attr"
            if idx_name not in obs:
                return f"error:index-col-'{idx_name}'-missing"
            if isinstance(obs[idx_name], h5py.Group):
                # Categorical index (categories/codes); cell_load can read either an
                # obs/_index dataset OR obs/_index/{categories,codes}, so a rename
                # still works -- but flag it so the operator can eyeball it.
                if dry_run:
                    return f"would-fix(categorical:{idx_name})"
                obs.move(idx_name, "_index")
                obs.attrs["_index"] = "_index"
                return f"fixed(categorical:{idx_name})"

            if dry_run:
                return f"would-fix({idx_name})"

            obs.move(idx_name, "_index")          # in-place HDF5 link rename, no data copy
            obs.attrs["_index"] = "_index"
            return f"fixed({idx_name})"
    except Exception as e:  # noqa: BLE001 - report, keep going across files
        return f"error:{type(e).__name__}:{e}"


def verify_file(path: str) -> str:
    """Read back the first few barcodes via obs/_index the way cell_load does."""
    try:
        with h5py.File(path, "r") as f:
            obs = f["obs"]
            node = obs["_index"]
            if isinstance(node, h5py.Group):  # categorical
                cats = node["categories"][:3]
                vals = [c.decode() if isinstance(c, bytes) else str(c) for c in cats]
            else:
                vals = [b.decode() if isinstance(b, bytes) else str(b) for b in node[:3]]
            return f"ok attr=_index sample={vals}"
    except Exception as e:  # noqa: BLE001
        return f"verify-error:{type(e).__name__}:{e}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*", help="specific h5ad files (default: all *.h5ad in --dir)")
    p.add_argument("--dir", default=DEFAULT_DIR, help="directory of h5ads when no files given")
    p.add_argument("--dry-run", action="store_true", help="report what would change, modify nothing")
    p.add_argument("--verify", action="store_true", help="after fixing, read back obs/_index per file")
    args = p.parse_args()

    files = args.files or sorted(glob.glob(os.path.join(args.dir, "*.h5ad")))
    if not files:
        print(f"No h5ad files found (dir={args.dir}).", file=sys.stderr)
        sys.exit(1)

    print(f"{'DRY-RUN: ' if args.dry_run else ''}Processing {len(files)} file(s)", flush=True)
    counts = {}
    for i, path in enumerate(files, 1):
        status = fix_file(path, args.dry_run)
        key = status.split("(")[0].split(":")[0]
        counts[key] = counts.get(key, 0) + 1
        line = f"[{i}/{len(files)}] {os.path.basename(path)}: {status}"
        if args.verify and status.startswith("fixed"):
            line += f"  ->  {verify_file(path)}"
        print(line, flush=True)

    summary = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"Done. {summary}", flush=True)
    if any(k.startswith("error") for k in counts):
        sys.exit(2)


if __name__ == "__main__":
    main()
