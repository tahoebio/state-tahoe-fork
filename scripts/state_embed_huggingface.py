import os
import logging
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from datasets import load_dataset
from omegaconf import OmegaConf as om, DictConfig
from state.emb.data.loader import VCIDatasetSentenceCollator
from state.emb.inference import Inference
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm.auto import tqdm

# === Logging Setup ===
log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
    level=logging.INFO,
)


class FilteredGenesCountsTahoe(Dataset):
    def __init__(self, source_dataset, reverse_token_id):
        self.source_dataset = source_dataset
        self.output_dim = len(reverse_token_id)
        self.reverse_token_id_tensor = torch.full(
            (max(reverse_token_id.keys()) + 1,), -1, dtype=torch.long
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


def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    dist.destroy_process_group()


def main(cfg: DictConfig) -> None:
    local_rank = setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    log.info(f"Rank {rank} started with local_rank {local_rank}...")

    ds = load_dataset(cfg.data.dataset_path, split=cfg.data.split, cache_dir=cfg.data.cache_dir).with_format("torch")
    gene_metadata = load_dataset(cfg.data.gene_metadata_path, cfg.data.gene_metadata_config)["train"].to_pandas()
    protein_embeds = torch.load(cfg.model.protein_embed_path, weights_only=False, map_location="cpu")

    valid_genes = list(protein_embeds.keys())
    global_pos = {g: i for i, g in enumerate(valid_genes)}
    gene_metadata["state_token_id"] = gene_metadata["gene_symbol"].apply(lambda g: global_pos.get(g, -1))
    reverse_token_id = dict(zip(gene_metadata["token_id"].values, gene_metadata.index.values))

    dataset = FilteredGenesCountsTahoe(source_dataset=ds, reverse_token_id=reverse_token_id)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)

    checkpoint_config = om.load(os.path.join(cfg.model.model_dir, "config.yaml"))
    checkpoint_config.model.batch_size = cfg.inference.batch_size

    collator = VCIDatasetSentenceCollator(
        checkpoint_config,
        valid_gene_mask={"tahoe-100M": gene_metadata["state_token_id"].values != -1},
        ds_emb_mapping_inference={"tahoe-100M": gene_metadata["state_token_id"].values},
        is_train=False,
        precision=torch.bfloat16,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.inference.batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=cfg.inference.num_workers,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=cfg.inference.prefetch_factor,
    )

    inference = Inference(cfg=checkpoint_config, protein_embeds=protein_embeds)
    inference.load_model(os.path.join(cfg.model.model_dir, cfg.model.checkpoint_file))
    model = inference.model.to(local_rank)
    if cfg.inference.compile:
        model = torch.compile(model, mode="default", fullgraph=True)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    os.makedirs(cfg.output.dir, exist_ok=True)
    schema = pa.schema([
        pa.field("dataset_index", pa.int32()),
        pa.field("state_embeddings", pa.list_(pa.float32(), 2048)),
    ])

    row_count, shard_idx, writer = 0, 0, None
    pbar = tqdm(total=len(sampler), desc=f"Rank {rank} writing", disable=(rank != 0))

    with torch.no_grad(), torch.amp.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
        for batch in dataloader:
            if writer is None:
                shard_path = os.path.join(cfg.output.dir, f"rank{rank}_state_{shard_idx:03d}.parquet")
                writer = pq.ParquetWriter(shard_path, schema, use_dictionary=True)

            _, _, _, emb, _ = model.module._compute_embedding_for_batch(batch)
            embeddings = emb.to("cpu").to(torch.float32).numpy()
            indices = batch[3]
            table = pa.Table.from_pydict({
                "dataset_index": indices,
                "state_embeddings": [list(row) for row in embeddings],
            }, schema=schema)

            writer.write_table(table)
            row_count += len(embeddings)
            pbar.update(len(embeddings))

            if row_count >= cfg.output.chunk_size:
                writer.close()
                writer = None
                row_count = 0
                shard_idx += 1

    if writer:
        writer.close()
    pbar.close()
    cleanup()


if __name__ == "__main__":
    import sys
    yaml_path = sys.argv[1]
    log.info(f"Loading configuration from {yaml_path}...")
    cfg = om.load(yaml_path)
    om.resolve(cfg)
    main(cfg)
    log.info("Script execution completed.")
