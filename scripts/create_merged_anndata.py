import logging
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
from polars import StringCache
from omegaconf import OmegaConf as om, DictConfig
from scipy import sparse
from tqdm import tqdm

# === Logging Setup ===
log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
    level=logging.INFO,
)

def load_hvg_mapping(token2hvg_path):
    log.info(f"Loading HVG mapping from: {token2hvg_path}")
    df = pd.read_parquet(token2hvg_path)
    df = df.sort_values('token_id').reset_index(drop=True)
    token_to_col_idx = {tid: i for i, tid in enumerate(df['token_id'])}
    gene_names = df['gene_symbol'].tolist()
    log.info(f"Loaded {len(token_to_col_idx)} HVG genes")
    return token_to_col_idx, gene_names

def save_chunk(hvg_matrix, mosaicfm_matrix, state_matrix, obs_data, gene_names, out_dir, idx, n_obs):
    log.info(f"Saving chunk {idx} with {n_obs} cells to {out_dir}")
    X_dummy = sparse.csr_matrix((n_obs, len(gene_names)))
    adata = ad.AnnData(X=X_dummy, obs=pd.DataFrame(obs_data[:n_obs]))
    adata.var_names = gene_names
    adata.var['gene_symbol'] = gene_names
    adata.obsm['X_hvg'] = hvg_matrix[:n_obs]
    adata.obsm['mosaicfm-70m-merged'] = mosaicfm_matrix[:n_obs]
    adata.obsm['state-SE-600M'] = state_matrix[:n_obs]
    adata.write_h5ad(out_dir / f"chunk_rank{cfg.rank:02d}_{idx:03d}.h5ad")

def main(cfg: DictConfig):
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Starting HVG conversion...")
    token_to_col_idx, gene_names = load_hvg_mapping(cfg.token2hvg_path)
    n_hvg_genes = len(gene_names)

    log.info("Opening parquet files lazily...")
    with StringCache():
        state_lf = pl.scan_parquet(cfg.state_path)
        mosaicfm_lf = pl.scan_parquet(cfg.mosaicfm_path).select([
            "BARCODE_SUB_LIB_ID", "mosaicfm-70m-merged"
        ])

        total_rows = state_lf.select(pl.len()).collect().item()
        log.info(f"Total rows to process: {total_rows:,}")

        rows_per_worker = total_rows // cfg.world_size
        start = cfg.rank * rows_per_worker
        end = total_rows if cfg.rank == cfg.world_size - 1 else (cfg.rank + 1) * rows_per_worker
        offset = start
        idx = 0

        pbar = tqdm(total=end - start, desc=f"Rank {cfg.rank} progress", unit="cells")

        log.info(f"Rank {cfg.rank} processing rows {start:,} to {end:,}...")
        while offset < end:
            t0 = time.time()
            rows_remaining = end - offset
            next_chunk_size = min(cfg.chunk_size, rows_remaining)

            hvg_matrix = np.zeros((next_chunk_size, n_hvg_genes), dtype=np.float32)
            mosaicfm_matrix = np.zeros((next_chunk_size, 512), dtype=np.float32)
            state_matrix = np.zeros((next_chunk_size, 2048), dtype=np.float32)
            obs_data = [None] * next_chunk_size

            state_batch = state_lf.slice(offset, next_chunk_size).collect()
            mosaic_batch = mosaicfm_lf.slice(offset, next_chunk_size).collect()

            for i, (s_row, m_row) in enumerate(zip(
                state_batch.iter_rows(named=True), mosaic_batch.iter_rows(named=True)
            )):
                assert s_row['BARCODE_SUB_LIB_ID'] == m_row['BARCODE_SUB_LIB_ID'], \
                    f"Mismatch: {s_row['BARCODE_SUB_LIB_ID']} != {m_row['BARCODE_SUB_LIB_ID']}"

                genes = s_row['genes']
                exprs = s_row['expressions']
                if exprs[0] < 0:
                    genes, exprs = genes[1:], exprs[1:]
                lib_size = sum(exprs)
                hvg_vec = np.zeros(n_hvg_genes, dtype=np.float32)
                for gene, expr in zip(genes, exprs):
                    if gene in token_to_col_idx:
                        hvg_vec[token_to_col_idx[gene]] = expr
                if lib_size > 0:
                    hvg_vec *= cfg.target_sum / lib_size

                obs_row = {k: s_row[k] for k in s_row.keys() if k not in ['genes', 'expressions', 'state_embeddings']}
                obs_row['library_size'] = lib_size

                hvg_matrix[i] = hvg_vec
                mosaicfm_matrix[i] = np.array(m_row['mosaicfm-70m-merged'], dtype=np.float32)
                state_matrix[i] = np.array(s_row['state_embeddings'], dtype=np.float32)
                obs_data[i] = obs_row

            save_chunk(hvg_matrix, mosaicfm_matrix, state_matrix, obs_data, gene_names, out_dir, idx, next_chunk_size)

            idx += 1
            offset += next_chunk_size
            pbar.update(next_chunk_size)
            max_chunks = cfg.get('max_chunks', None)

            if max_chunks is not None and idx >= max_chunks:
                log.info("Reached max_chunks limit — exiting early.")
                break

            t1 = time.time()
            log.info(f"Chunk {idx} processed in {t1 - t0:.2f}s")

        pbar.close()
        log.info(f"Rank {cfg.rank}: All chunks written.")

if __name__ == "__main__":
    yaml_path = sys.argv[1]
    cfg = om.load(yaml_path)
    om.resolve(cfg)

    if len(sys.argv) > 2 and sys.argv[2].startswith("rank="):
        cfg.rank = int(sys.argv[2].split("=")[1])

    main(cfg)
    log.info("Script execution completed.")
