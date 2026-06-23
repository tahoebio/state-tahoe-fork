#!/usr/bin/env python
"""
soma_to_state_h5ad.py — Export the Caspian TileDB-SOMA drug screen to per-cell-line
.h5ad files in the format Arc's upstream STATE (`state tx train`) consumes.

This is the Caspian analogue of the fork's create_merged_anndata_by_plate.py: it
produces one h5ad per cell line, each with depth-normalized HVG expression in
.obsm['X_hvg']. The `state` PertSets datamodule (cell_load) reads .obsm['X_hvg']
verbatim — there is no transform hook — so whatever normalization you want the model
to train on must be baked in here. The h5ad files are fold-independent: every
train/val/test split is just a different TOML pointing at the same output directory.

Default (linear, matches the Tahoe runs):
  X_hvg[gene] = raw_count[gene] / total_counts(cell) * target_sum
With --log1p (train STATE in log1p space; also sets uns/log1p so cell_load detects it):
  X_hvg[gene] = log1p(raw_count[gene] / total_counts(cell) * target_sum)
  - raw_count: the 2000-HVG subset of the raw UMI counts (X layer "data")
  - total_counts: the cell's full-genome library size, read cheaply from the obs
    column `rna:tscp_count` (verified == full row sum of X)
  - target_sum: Caspian median library size (default 2183, per Rhaister)

.X is left as an empty sparse placeholder (n_cells x 2000); the real data lives in
.obsm['X_hvg'], which STATE row-indexes from the h5 file (hence dense, not sparse).

obs columns written (the values STATE keys on):
  drugname_drugconc  -> pert_col       (control auto-detected as "[('DMSO_TF', 0.0, 'uM')]")
  Cell_ID_Cellosaur  -> cell_type_key  (CVCL id, matches the split TOML keys)
  plate              -> batch_col      (for basal_mapping_strategy=batch -> same-plate controls)

Run with an env that has tiledbsoma + anndata (e.g. Rhaister's .venv):
  /home/umair/Rhaister/.venv/bin/python soma_to_state_h5ad.py --out-dir <dir>
"""
import argparse
import json
import os
import time

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import tiledbsoma as soma

DEFAULT_QC = "pass_filter == 'full' and attr(\"DROPLET.TYPE\") == 'SNG'"


def resolve_hvg(exp, measurement, hvg_genes, gene_name_col="gene_name"):
    """Map the static HVG gene-name list to soma_joinids.

    Duplicate gene symbols (5 in Caspian) are resolved to their lowest joinid,
    mirroring Rhaister's _static_var_subset. Returns:
      sorted_join  : HVG joinids in ascending order (what SOMA returns columns in)
      json_to_col  : for each gene in hvg_genes order, its column index in the
                     joinid-sorted query result (so we can reorder X to list order)
    """
    var = exp.ms[measurement].var.read(column_names=["soma_joinid", gene_name_col]).concat().to_pandas()
    hset = set(hvg_genes)
    seen = {}
    for jid, gn in zip(var["soma_joinid"], var[gene_name_col]):
        gn = str(gn)
        if gn in hset and gn not in seen:
            seen[gn] = int(jid)
    missing = [g for g in hvg_genes if g not in seen]
    if missing:
        raise SystemExit(f"{len(missing)} HVG genes absent from var['{gene_name_col}'], e.g. {missing[:5]}")
    sorted_join = sorted(seen[g] for g in hvg_genes)
    col_of = {j: k for k, j in enumerate(sorted_join)}
    json_to_col = np.array([col_of[seen[g]] for g in hvg_genes], dtype=np.int64)
    return sorted_join, json_to_col


def list_cell_lines(exp, cl_col, qc):
    obs = exp.obs.read(column_names=[cl_col], value_filter=qc).concat().to_pandas()
    return sorted(obs[cl_col].astype(str).unique())


def export_line(exp, cl, args, sorted_join, json_to_col, hvg_names, out_path):
    cl_col, treat_col, plate_col, lib_col = (
        args.cell_line_col, args.treatment_col, args.plate_col, args.libsize_col,
    )
    obs = (
        exp.obs.read(
            column_names=["soma_joinid", "obs_id", treat_col, plate_col, lib_col, cl_col],
            value_filter=f'{cl_col} == "{cl}" and {args.qc_filter}',
        )
        .concat()
        .to_pandas()
        .set_index("soma_joinid")
    )
    if args.limit_cells:
        obs = obs.iloc[: args.limit_cells]
    n = len(obs)
    if n == 0:
        return 0
    jids = obs.index.astype(int).to_numpy()
    n_hvg = len(hvg_names)

    X = np.zeros((n, n_hvg), dtype=np.float32)
    meta_chunks = []
    off = 0
    for s in range(0, n, args.chunk_size):
        cj = jids[s : s + args.chunk_size].tolist()
        with exp.axis_query(
            args.measurement,
            obs_query=soma.AxisQuery(coords=(cj,)),
            var_query=soma.AxisQuery(coords=(sorted_join,)),
        ) as q:
            a = q.to_anndata(X_name=args.x_layer)
        rj = a.obs["soma_joinid"].astype(int).to_numpy()
        counts = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
        counts = counts.astype(np.float32)[:, json_to_col]  # reorder to HVG-list order
        lib = obs.loc[rj, lib_col].to_numpy().astype(np.float32)
        lib = np.where(lib > 0, lib, 1.0)
        counts *= (args.target_sum / lib)[:, None]
        if args.log1p:
            # log1p(depth-normalized) — the PertSets model reads .obsm['X_hvg']
            # verbatim (no transform hook), so log1p must be baked in here.
            np.log1p(counts, out=counts)
        m = counts.shape[0]
        X[off : off + m] = counts
        meta_chunks.append(obs.loc[rj])
        off += m
    assert off == n, f"row count mismatch: filled {off} of {n}"

    meta = pd.concat(meta_chunks)
    out_obs = pd.DataFrame(
        {
            treat_col: meta[treat_col].astype(str).to_numpy(),
            cl_col: meta[cl_col].astype(str).to_numpy(),
            plate_col: meta[plate_col].astype(str).to_numpy(),
        },
        # Leave the index UNNAMED so AnnData writes the cell barcodes to obs/_index
        # (and sets obs.attrs["_index"] = "_index"). cell_load's _load_cell_barcodes
        # reads the literal obs/_index key; a named index (e.g. "obs_id") lands at
        # obs/obs_id instead, which cell_load can't find -> it falls back to generic
        # "cell_000000" barcodes. Unnamed keeps the real obs_id values discoverable.
        index=pd.Index(meta["obs_id"].astype(str).to_numpy()),
    )
    var = pd.DataFrame(index=pd.Index(hvg_names, name="gene_name"))
    A = ad.AnnData(X=sp.csr_matrix((n, n_hvg), dtype=np.float32), obs=out_obs, var=var)
    A.obsm["X_hvg"] = X
    if args.log1p:
        # Signals cell_load to set is_log1p=True (matches the baked-in transform);
        # avoids the "is_log1p enabled but no uns/log1p detected" warning and keeps
        # downsampling/eval math consistent.
        A.uns["log1p"] = {"base": None}
    A.write_h5ad(out_path)
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--soma-uri", default="/nvme-shared/datasets/caspian/soma/concatenated_soma/")
    p.add_argument("--out-dir", default="/nvme-shared/datasets/caspian/state_input/")
    p.add_argument("--hvg-json", default="/home/umair/Rhaister/splits/caspian/caspian_hvg_2k_genes.json")
    p.add_argument("--measurement", default="rna")
    p.add_argument("--x-layer", default="data")
    p.add_argument("--cell-line-col", default="Cell_ID_Cellosaur")
    p.add_argument("--treatment-col", default="drugname_drugconc")
    p.add_argument("--plate-col", default="plate")
    p.add_argument("--libsize-col", default="rna:tscp_count")
    p.add_argument("--gene-name-col", default="gene_name")
    p.add_argument("--target-sum", type=float, default=2183.0)
    p.add_argument("--log1p", action="store_true",
                   help="Store log1p(depth-normalized) in .obsm['X_hvg'] and set uns/log1p "
                        "(train STATE in log1p space). Default: linear depth-normalized.")
    p.add_argument("--qc-filter", default=DEFAULT_QC)
    p.add_argument("--chunk-size", type=int, default=100_000, help="cells per X fetch (memory control)")
    p.add_argument("--cell-lines", nargs="*", default=None, help="subset of cell lines (default: all)")
    p.add_argument("--limit-cells", type=int, default=None, help="cap cells per line (smoke test only)")
    p.add_argument("--overwrite", action="store_true", help="re-export lines whose h5ad already exists")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    hvg_names = json.load(open(args.hvg_json))
    print(f"HVG genes: {len(hvg_names)}  target_sum={args.target_sum}  out={args.out_dir}", flush=True)

    exp = soma.open(args.soma_uri)
    sorted_join, json_to_col = resolve_hvg(exp, args.measurement, hvg_names, args.gene_name_col)
    print(f"Resolved {len(sorted_join)} HVG joinids", flush=True)

    cell_lines = args.cell_lines or list_cell_lines(exp, args.cell_line_col, args.qc_filter)
    print(f"Exporting {len(cell_lines)} cell lines", flush=True)

    manifest = {}
    for i, cl in enumerate(cell_lines, 1):
        out_path = os.path.join(args.out_dir, f"{cl}.h5ad")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[{i}/{len(cell_lines)}] {cl}: exists, skip", flush=True)
            continue
        t0 = time.time()
        ncells = export_line(exp, cl, args, sorted_join, json_to_col, hvg_names, out_path)
        manifest[cl] = ncells
        sz = os.path.getsize(out_path) / 1e9 if os.path.exists(out_path) else 0
        print(f"[{i}/{len(cell_lines)}] {cl}: {ncells:,} cells  {sz:.1f} GB  {time.time()-t0:.0f}s", flush=True)

    if manifest:
        with open(os.path.join(args.out_dir, "_export_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
    print(f"Done. Exported {sum(manifest.values()):,} cells across {len(manifest)} lines.", flush=True)


if __name__ == "__main__":
    main()
