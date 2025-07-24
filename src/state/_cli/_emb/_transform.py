import argparse as ap
from typing import List, Optional
import os

def find_h5ad_files(
    directory: str,
    ignore_subdirs: Optional[List[str]] = None,
) -> List[str]:
    """Recursively search for all '.h5ad' files in a given directory while
    ignoring specified subdirectories."""
    h5ad_files = []
    ignore_subdirs = ignore_subdirs or []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in ignore_subdirs]
        for file in files:
            if file.endswith(".h5ad") or file.endswith(".h5ad.gz"):
                h5ad_files.append(os.path.join(root, file))
    return h5ad_files


def add_arguments_transform(parser: ap.ArgumentParser):
    """Add arguments for state embedding CLI."""
    parser.add_argument("--model-folder", required=True, help="Path to the model checkpoint folder")
    parser.add_argument("--checkpoint", required=False, help="Path to the specific model checkpoint")
    parser.add_argument("--input", required=True, help="Path to input anndata file (h5ad) or directory")
    parser.add_argument("--output", required=True, help="Path to output embedded anndata file (h5ad) or directory")
    parser.add_argument("--embed-key", default="X_state", help="Name of key to store embeddings")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for encoding")


def run_emb_transform(args: ap.ArgumentParser):
    """
    Compute embeddings for an input anndata file or all files in a directory
    using a pre-trained VCI model checkpoint.
    """
    import glob
    import logging

    import torch
    from omegaconf import OmegaConf

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    from ...emb.inference import Inference

    # Locate checkpoint
    model_files = glob.glob(os.path.join(args.model_folder, "*.ckpt"))
    if not model_files:
        logger.error(f"No model checkpoint found in {args.model_folder}")
        raise FileNotFoundError(f"No model checkpoint found in {args.model_folder}")
    if not args.checkpoint:
        args.checkpoint = model_files[-1]
    logger.info(f"Using model checkpoint: {args.checkpoint}")

    # Load protein embeddings and config
    logger.info("Creating inference object")
    embedding_file = os.path.join(args.model_folder, "protein_embeddings.pt")
    protein_embeds = torch.load(embedding_file, weights_only=False, map_location="cpu")

    config_file = os.path.join(args.model_folder, "config.yaml")
    conf = OmegaConf.load(config_file)

    inferer = Inference(cfg=conf, protein_embeds=protein_embeds)

    # Load model weights
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    inferer.load_model(args.checkpoint)

    # Handle directory vs single file input
    if os.path.isdir(args.input):
        # Ensure output is a directory
        if not os.path.exists(args.output):
            os.makedirs(args.output, exist_ok=True)

        h5ad_files = find_h5ad_files(args.input)
        for input_file in h5ad_files:
            # Build mirrored subdir under output
            rel_path = os.path.relpath(input_file, args.input)
            rel_dir = os.path.dirname(rel_path)
            target_dir = os.path.join(args.output, rel_dir) if rel_dir else args.output
            os.makedirs(target_dir, exist_ok=True)

            # Derive output filename with '_state_embs' suffix
            filename = os.path.basename(input_file)
            if filename.endswith(".h5ad.gz"):
                stem = filename[:-len(".h5ad.gz")]
                ext = ".h5ad.gz"
            else:
                stem, ext = os.path.splitext(filename)
            output_file = os.path.join(target_dir, f"{stem}_state_embs{ext}")

            logger.info(f"Computing embeddings for {input_file} -> {output_file}")
            inferer.encode_adata(
                input_adata_path=input_file,
                output_adata_path=output_file,
                emb_key=args.embed_key,
                batch_size=args.batch_size,
            )
    else:
        # Single file case
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")

        logger.info(f"Computing embeddings for {args.input}")
        logger.info(f"Output will be saved to {args.output}")
        inferer.encode_adata(
            input_adata_path=args.input,
            output_adata_path=args.output,
            emb_key=args.embed_key,
            batch_size=args.batch_size,
        )

    logger.info("Embedding computation completed successfully!")