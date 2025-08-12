import os
from pathlib import Path
import logging

import anndata as ad
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

# Logging setup
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
log = logging.getLogger(__name__)


def build_sample_to_dose_map():
    log.info("Loading sample metadata from Hugging Face...")
    sample_ds = load_dataset("tahoebio/Tahoe-100M", "sample_metadata", split="train").to_pandas()
    sample_to_dose = dict(zip(sample_ds["sample"], sample_ds["drugname_drugconc"]))
    log.info(f"Loaded {len(sample_to_dose)} sample→dose mappings")
    return sample_to_dose


def update_obs_with_drugdose(h5ad_path: Path, sample_to_dose: dict):
    try:
        adata = ad.read_h5ad(h5ad_path, backed=None)

        if "sample" not in adata.obs:
            log.warning(f"'sample' column missing in {h5ad_path.name}, skipping.")
            return False

        adata.obs["drugname_drugconc"] = adata.obs["sample"].map(sample_to_dose)

        if adata.obs["drugname_drugconc"].isnull().any():
            log.warning(f"Null drugdose values in {h5ad_path.name}, some samples may be unmapped.")

        h5ad_path.unlink()  # Safely remove the file first
        adata.write_h5ad(str(h5ad_path))  # Save updated AnnData
        return True

    except Exception as e:
        log.error(f"Failed to update {h5ad_path.name}: {e}")
        return False


def update_all_chunks(folder: Path):
    log.info(f"Scanning for .h5ad files in: {folder}")
    h5ad_files = sorted(folder.rglob("*.h5ad"))
    log.info(f"Found {len(h5ad_files)} files.")

    sample_to_dose = build_sample_to_dose_map()

    success = 0
    with tqdm(total=len(h5ad_files), desc="Updating .h5ad files") as pbar:
        for file in h5ad_files:
            log.info(f"Updating: {file}")
            ok = update_obs_with_drugdose(file, sample_to_dose)
            if ok:
                log.info(f"✓ Updated {file.name}")
                success += 1
            pbar.update(1)

    log.info(f"✓ Successfully updated {success}/{len(h5ad_files)} files.")

# Path to h5ad chunks folder
folder = Path("/tahoe/drive_3/ANALYSIS/analysis_190/Data/state_input_merged")

update_all_chunks(folder)

