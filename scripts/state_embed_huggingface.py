from state.emb.inference import Inference
from state.emb.data.loader import VCIDatasetSentenceCollator

from omegaconf import OmegaConf

import h5py as h5
import anndata
import torch
import os
import numpy as np
from tqdm.auto import tqdm
import pyarrow as pa
from torch.utils.data import DataLoader
import pyarrow.parquet as pq

from datasets import load_dataset
from torch.utils.data import Dataset



class FilteredGenesCountsTahoe(Dataset):
    def __init__(self, source_dataset, reverse_token_id):
        self.source_dataset = source_dataset

        # Build vectorized mapping tensor
        self.reverse_token_id_tensor = torch.full(
            (max(reverse_token_id.keys()) + 1,),
            -1,
            dtype=torch.long
        )
        for k, v in reverse_token_id.items():
            self.reverse_token_id_tensor[k] = v

        self.output_dim = len(reverse_token_id)

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

ds = load_dataset(
    "/tahoe/mosaicfm/datasets/Tahoe-100M-HF",
    split="train",
    cache_dir="/tahoe/mosaicfm/datasets/HF_cache",
)
model_folder = "/home/shreshth/state-tahoe-fork/SE-600M"
checkpoint_name = "se600m_epoch4.ckpt"

ds = ds.with_format("torch")
gene_metadata = load_dataset("tahoebio/Tahoe-100M","gene_metadata")["train"].to_pandas()
protein_embeds = torch.load(f"{model_folder}/protein_embeddings.pt", weights_only=False, map_location="cpu")
valid_genes_list = list(protein_embeds.keys())
global_pos = {g: i for i, g in enumerate(valid_genes_list)}
gene_metadata["state_token_id"] = gene_metadata["gene_symbol"].apply(lambda gene_name: global_pos.get(gene_name,-1))

reverse_token_id = dict(zip(gene_metadata["token_id"].values, gene_metadata.index.values))

new_ds = FilteredGenesCountsTahoe(ds,
                                  reverse_token_id)

model_config = OmegaConf.load(f"{model_folder}/config.yaml")
model_config.model.batch_size = 256

collator = VCIDatasetSentenceCollator(model_config,
                                     valid_gene_mask = {"tahoe-100M":gene_metadata["state_token_id"].values != -1},
                                     ds_emb_mapping_inference={"tahoe-100M":gene_metadata["state_token_id"].values},
                                     is_train=False,
                                     precision=torch.bfloat16,
                                     )

dataloader = DataLoader(
    new_ds,
    batch_size=model_config.model.batch_size,
    shuffle=False,
    collate_fn=collator,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    prefetch_factor=4,
)

inference_class = Inference(cfg=model_config, protein_embeds=protein_embeds)

inference_class.load_model(os.path.join(model_folder, checkpoint_name))
model = torch.compile(inference_class.model)
device = next(model.parameters()).device

schema = pa.schema(
        [
            pa.field("dataset_index", pa.int32()),
            pa.field("state_embeddings", pa.list_(pa.float32(), 2048)),
        ],
    )
output_dir = "/tahoe/mosaicfm/datasets/state_embeddings_tahoe_100m_w_idx"
os.makedirs(output_dir, exist_ok=True)
total_rows = len(new_ds)
row_count = 0
shard_idx = 0
writer = None
pbar = tqdm(total=total_rows, desc="Embedding & writing")

with torch.no_grad(), torch.amp.autocast(
        enabled=True,
        dtype=torch.bfloat16,
        device_type=device.type,
):
    for batch in dataloader:

        # Rotate to a new ParquetWriter if starting a shard
        if writer is None:
            shard_path = os.path.join(
                output_dir,
                f"state_{shard_idx:03d}.parquet",
            )
            writer = pq.ParquetWriter(shard_path, schema, use_dictionary=True)

        _, _, _, emb, _ = model._compute_embedding_for_batch(batch)
        embeddings = emb.to("cpu").to(torch.float32).numpy()
        bs = embeddings.shape[0]
        dataset_indices = batch[3]
        table = pa.Table.from_pydict(
            {"dataset_index": dataset_indices,
             "state_embeddings": [list(r) for r in embeddings],
             },
            schema=schema,
        )
        writer.write_table(table)
        row_count += bs
        pbar.update(bs)

        # If chunk size reached, close and advance shard
        if row_count >= 100000:
            writer.close()
            writer = None
            row_count = 0
            shard_idx += 1
# Final close
if writer:
    writer.close()
pbar.close()