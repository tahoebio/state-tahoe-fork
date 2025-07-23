import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from state.emb.data.loader import VCIDatasetSentenceCollator
from state.emb.inference import Inference
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# === Arguments ===
DATASET_PATH = "/tahoe/mosaicfm/datasets/Tahoe-100M-HF"
DATASET_SPLIT = "train"
CACHE_DIR = "/tahoe/mosaicfm/datasets/HF_cache"
MODEL_DIR = "/home/shreshth/state-tahoe-fork/SE-600M"
CHECKPOINT_FILE = "se600m_epoch4.ckpt"
OUTPUT_DIR = "/tahoe/mosaicfm/datasets/state_embeddings_tahoe_100m_w_idx"
BATCH_SIZE = 256
NUM_WORKERS = 8
PREFETCH_FACTOR = 4
CHUNK_SIZE = 100_000
COMPILE=False


class FilteredGenesCountsTahoe(Dataset):
    def __init__(self, source_dataset, reverse_token_id):
        self.source_dataset = source_dataset
        self.output_dim = len(reverse_token_id)

        self.reverse_token_id_tensor = torch.full(
            (max(reverse_token_id.keys()) + 1,),
            -1,
            dtype=torch.long
        )
        for k, v in reverse_token_id.items():
            self.reverse_token_id_tensor[k] = v

    def __len__(self):
        return len(self.source_dataset)

    def __getitem__(self, idx):
        row = self.source_dataset[idx]
        genes = row["genes"][1:]
        exprs = row["expressions"][1:]

        mapped = self.reverse_token_id_tensor[genes]
        valid = mapped >= 0

        exp = torch.zeros(self.output_dim, dtype=torch.float32)
        exp[mapped[valid]] = exprs[valid]

        return exp.unsqueeze(0), idx, "tahoe-100M", 0


# === Load Data & Metadata ===
ds = load_dataset(DATASET_PATH, split=DATASET_SPLIT, cache_dir=CACHE_DIR).with_format("torch")
gene_metadata = load_dataset("tahoebio/Tahoe-100M", "gene_metadata")["train"].to_pandas()
protein_embeds = torch.load(f"{MODEL_DIR}/protein_embeddings.pt", weights_only=False, map_location="cpu")

valid_genes = list(protein_embeds.keys())
global_pos = {g: i for i, g in enumerate(valid_genes)}
gene_metadata["state_token_id"] = gene_metadata["gene_symbol"].apply(lambda g: global_pos.get(g, -1))
reverse_token_id = dict(zip(gene_metadata["token_id"].values, gene_metadata.index.values))


# === Model & DataLoader Setup ===
dataset = FilteredGenesCountsTahoe(source_dataset=ds, reverse_token_id=reverse_token_id)
config = OmegaConf.load(f"{MODEL_DIR}/config.yaml")
config.model.batch_size = BATCH_SIZE # Override batch size for inference

collator = VCIDatasetSentenceCollator(
    config,
    valid_gene_mask={"tahoe-100M": gene_metadata["state_token_id"].values != -1},
    ds_emb_mapping_inference={"tahoe-100M": gene_metadata["state_token_id"].values},
    is_train=False,
    precision=torch.bfloat16,
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collator,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=PREFETCH_FACTOR,
)

inference = Inference(cfg=config, protein_embeds=protein_embeds)
inference.load_model(os.path.join(MODEL_DIR, CHECKPOINT_FILE))
model = inference.model
if COMPILE:
    model= torch.compile(model, mode="default", fullgraph=True)
device = next(model.parameters()).device

# === Output Schema ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

schema = pa.schema([
    pa.field("dataset_index", pa.int32()),
    pa.field("state_embeddings", pa.list_(pa.float32(), 2048)),
])

# === Inference & Write Loop ===
row_count = 0
shard_idx = 0
writer = None
pbar = tqdm(total=len(dataset), desc="Embedding & writing")

with torch.no_grad(), torch.amp.autocast(enabled=True, dtype=torch.bfloat16, device_type=device.type):
    for batch in dataloader:
        if writer is None:
            shard_path = os.path.join(OUTPUT_DIR, f"state_{shard_idx:03d}.parquet")
            writer = pq.ParquetWriter(shard_path, schema, use_dictionary=True)

        _, _, _, emb, _ = model._compute_embedding_for_batch(batch)
        embeddings = emb.to("cpu").to(torch.float32).numpy()
        indices = batch[3]
        table = pa.Table.from_pydict({
            "dataset_index": indices,
            "state_embeddings": [list(row) for row in embeddings],
        }, schema=schema)

        writer.write_table(table)
        row_count += len(embeddings)
        pbar.update(len(embeddings))

        if row_count >= CHUNK_SIZE:
            writer.close()
            writer = None
            row_count = 0
            shard_idx += 1

if writer:
    writer.close()
pbar.close()
