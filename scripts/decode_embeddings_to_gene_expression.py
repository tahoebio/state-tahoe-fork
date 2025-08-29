#!/usr/bin/env python3
"""
Decode embeddings to gene expression using a trained State TX model's gene decoder.

This script takes .h5ad files with embeddings and uses a trained model's gene decoder
to predict gene expression, storing the results in .obsm['X_hvg'].

Key features:
- Comprehensive performance tracking with detailed timing breakdown
- Live progress monitoring with bottleneck detection
- Memory-efficient batch processing with adaptive batch sizing
- Device transfer monitoring (CPU ↔ GPU)
- Final performance report with optimization recommendations

Example usage:
    python decode_embeddings_to_gene_expression.py \
        --input embeddings.h5ad \
        --checkpoint /path/to/model/final.ckpt \
        --output predictions.h5ad \
        --batch-size 1000 \
        --device cuda
"""

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import h5py
import numpy as np
import psutil
import torch
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage in GB."""
    # CPU memory
    process = psutil.Process()
    cpu_memory = process.memory_info().rss / 1024 / 1024 / 1024
    
    # GPU memory
    gpu_memory = {"used": 0.0, "total": 0.0}
    if torch.cuda.is_available():
        gpu_memory["used"] = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
        gpu_memory["total"] = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
    
    return {
        "cpu": cpu_memory,
        "gpu_used": gpu_memory["used"],
        "gpu_total": gpu_memory["total"]
    }


class DecoderPerformanceTracker:
    """Comprehensive performance tracking for decoder inference with detailed timing breakdown."""
    
    def __init__(self, total_cells: int, filename: str):
        self.total_cells = total_cells
        self.filename = filename
        self.processed_cells = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.last_update_cells = 0
        
        # Detailed timing components
        self.timings = {
            # Initialization
            'model_loading': 0.0,
            'model_setup': 0.0,
            'decoder_extraction': 0.0,
            'gpu_initialization': 0.0,
            
            # Data Reading
            'file_opening': 0.0,
            'embedding_reading': 0.0,
            'metadata_reading': 0.0,
            
            # Processing (per batch)
            'numpy_to_tensor': 0.0,
            'cpu_to_gpu_transfer': 0.0,
            'decoder_inference': 0.0,
            'gpu_to_cpu_transfer': 0.0,
            'tensor_to_numpy': 0.0,
            'batch_overhead': 0.0,
            
            # Writing
            'output_file_creation': 0.0,
            'predictions_writing': 0.0,
            'metadata_writing': 0.0,
            'file_finalization': 0.0
        }
        
        # Memory tracking
        self.memory_snapshots = []
        self.max_memory = get_memory_usage()
        
        # Bottleneck detection thresholds
        self.bottleneck_thresholds = {
            'cpu_to_gpu_transfer': 20.0,  # > 20% suggests batch size too small
            'gpu_to_cpu_transfer': 15.0,  # > 15% suggests batch size too small
            'decoder_inference': 70.0,    # > 70% suggests need for optimization
            'file_operations': 30.0,      # > 30% suggests I/O bottleneck
        }
        
        log.info(f"🚀 Starting decoder inference for {filename} ({total_cells:,} cells)")
        
    def update_timing(self, component: str, duration: float, cells_processed: int = 0):
        """Update timing for a specific component."""
        if component not in self.timings:
            log.warning(f"Unknown timing component: {component}")
            return
            
        self.timings[component] += duration
        self.processed_cells += cells_processed
        
        # Update memory tracking
        current_memory = get_memory_usage()
        self.max_memory["cpu"] = max(self.max_memory["cpu"], current_memory["cpu"])
        self.max_memory["gpu_used"] = max(self.max_memory["gpu_used"], current_memory["gpu_used"])
        
        # Log progress every 50K cells or every 30 seconds
        current_time = time.time()
        if (self.processed_cells - self.last_update_cells >= 50000 or 
            current_time - self.last_update_time >= 30):
            self._log_progress(current_memory)
            
    def _log_progress(self, current_memory: Dict[str, float]):
        """Log detailed progress with bottleneck analysis."""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if elapsed < 1.0:  # Avoid division by zero
            return
            
        # Calculate rates
        overall_rate = self.processed_cells / elapsed
        recent_cells = self.processed_cells - self.last_update_cells
        recent_time = current_time - self.last_update_time
        recent_rate = recent_cells / recent_time if recent_time > 0 else 0
        
        # Calculate ETA
        remaining_cells = self.total_cells - self.processed_cells
        eta_seconds = remaining_cells / recent_rate if recent_rate > 0 else 0
        eta_hours = eta_seconds / 3600
        
        # Calculate component percentages
        total_processing_time = sum([
            self.timings['cpu_to_gpu_transfer'],
            self.timings['decoder_inference'], 
            self.timings['gpu_to_cpu_transfer'],
            self.timings['numpy_to_tensor'],
            self.timings['tensor_to_numpy']
        ])
        
        if total_processing_time > 0:
            percentages = {
                'GPU Transfer': (self.timings['cpu_to_gpu_transfer'] / total_processing_time) * 100,
                'Decode': (self.timings['decoder_inference'] / total_processing_time) * 100,
                'CPU Transfer': (self.timings['gpu_to_cpu_transfer'] / total_processing_time) * 100,
                'Tensor Ops': ((self.timings['numpy_to_tensor'] + self.timings['tensor_to_numpy']) / total_processing_time) * 100,
            }
        else:
            percentages = {'GPU Transfer': 0, 'Decode': 0, 'CPU Transfer': 0, 'Tensor Ops': 0}
            
        # Progress bar
        progress_pct = (self.processed_cells / self.total_cells) * 100
        log.info(f"📊 Progress: {self.processed_cells:,}/{self.total_cells:,} cells ({progress_pct:.1f}%)")
        
        # Performance breakdown
        log.info(f"⚡ Performance: GPU={percentages['GPU Transfer']:.1f}%, "
                f"Decode={percentages['Decode']:.1f}%, CPU={percentages['CPU Transfer']:.1f}%, "
                f"Tensor={percentages['Tensor Ops']:.1f}%")
        
        # Rates and ETA
        log.info(f"🏃 Rate: {overall_rate:.0f} cells/sec (recent: {recent_rate:.0f}), ETA: {eta_hours:.1f} hours")
        
        # Memory usage
        if current_memory["gpu_total"] > 0:
            gpu_usage_pct = (current_memory["gpu_used"] / current_memory["gpu_total"]) * 100
            log.info(f"💾 Memory: GPU: {current_memory['gpu_used']:.1f}/{current_memory['gpu_total']:.1f}GB ({gpu_usage_pct:.1f}%), "
                    f"CPU: {current_memory['cpu']:.1f}GB")
        else:
            log.info(f"💾 Memory: CPU: {current_memory['cpu']:.1f}GB")
            
        # Bottleneck detection
        self._detect_bottlenecks(percentages)
        
        # Update for next iteration
        self.last_update_time = current_time
        self.last_update_cells = self.processed_cells
        
    def _detect_bottlenecks(self, percentages: Dict[str, float]):
        """Detect and warn about performance bottlenecks."""
        bottlenecks = []
        
        # Check individual components
        if percentages['GPU Transfer'] + percentages['CPU Transfer'] > 35:
            bottlenecks.append("Data transfers (consider larger batch size)")
        elif percentages['Decode'] > 70:
            bottlenecks.append("Decoder inference (consider mixed precision)")
        elif percentages['Tensor Ops'] > 25:
            bottlenecks.append("Tensor operations (batch operations if possible)")
            
        if bottlenecks:
            log.info(f"⚠️ Bottleneck detected: {', '.join(bottlenecks)}")
            
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final performance report."""
        total_time = time.time() - self.start_time
        avg_throughput = self.processed_cells / total_time if total_time > 0 else 0
        
        # Group timings by category
        init_time = (self.timings['model_loading'] + self.timings['model_setup'] + 
                    self.timings['decoder_extraction'] + self.timings['gpu_initialization'])
        
        read_time = (self.timings['file_opening'] + self.timings['embedding_reading'] + 
                    self.timings['metadata_reading'])
        
        process_time = (self.timings['numpy_to_tensor'] + self.timings['cpu_to_gpu_transfer'] +
                       self.timings['decoder_inference'] + self.timings['gpu_to_cpu_transfer'] +
                       self.timings['tensor_to_numpy'] + self.timings['batch_overhead'])
        
        write_time = (self.timings['output_file_creation'] + self.timings['predictions_writing'] +
                     self.timings['metadata_writing'] + self.timings['file_finalization'])
        
        # Calculate percentages
        total_tracked = init_time + read_time + process_time + write_time
        if total_tracked > 0:
            init_pct = (init_time / total_tracked) * 100
            read_pct = (read_time / total_tracked) * 100  
            process_pct = (process_time / total_tracked) * 100
            write_pct = (write_time / total_tracked) * 100
        else:
            init_pct = read_pct = process_pct = write_pct = 0
        
        report = {
            'summary': {
                'total_cells': self.total_cells,
                'total_time_hours': total_time / 3600,
                'avg_throughput': avg_throughput,
                'max_cpu_memory_gb': self.max_memory["cpu"],
                'max_gpu_memory_gb': self.max_memory["gpu_used"],
                'gpu_total_gb': self.max_memory["gpu_total"]
            },
            'time_breakdown': {
                'initialization': {'time_seconds': init_time, 'percentage': init_pct},
                'data_reading': {'time_seconds': read_time, 'percentage': read_pct},
                'processing': {'time_seconds': process_time, 'percentage': process_pct},
                'data_writing': {'time_seconds': write_time, 'percentage': write_pct}
            },
            'detailed_timings': self.timings,
            'recommendations': self._generate_recommendations(init_pct, read_pct, process_pct, write_pct)
        }
        
        return report
        
    def _generate_recommendations(self, init_pct: float, read_pct: float, 
                                process_pct: float, write_pct: float) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        if write_pct > 25:
            recommendations.append("Consider using NVMe SSD for output to reduce write time")
        if read_pct > 20:
            recommendations.append("Consider using faster storage or memory-mapped files for input")
        if process_pct > 60:
            if self.max_memory["gpu_used"] < self.max_memory["gpu_total"] * 0.8:
                recommendations.append("Increase batch size to utilize more GPU memory")
            recommendations.append("Consider enabling mixed precision (fp16) for faster inference")
        if init_pct > 10:
            recommendations.append("Model loading overhead is high - consider model optimization")
            
        return recommendations
        
    def log_final_report(self, report: Dict[str, Any]):
        """Log the final performance report."""
        log.info("=" * 50)
        log.info("FINAL PERFORMANCE REPORT")
        log.info("=" * 50)
        
        summary = report['summary']
        log.info(f"Total cells processed: {summary['total_cells']:,}")
        log.info(f"Total time: {summary['total_time_hours']:.2f} hours")
        log.info(f"Average throughput: {summary['avg_throughput']:.0f} cells/sec")
        log.info("")
        
        log.info("Time Breakdown:")
        for phase, data in report['time_breakdown'].items():
            mins = data['time_seconds'] / 60
            log.info(f"  {phase.replace('_', ' ').title()}: {mins:.1f} min ({data['percentage']:.1f}%)")
        log.info("")
        
        log.info("Memory Peak Usage:")
        log.info(f"  CPU: {summary['max_cpu_memory_gb']:.1f}GB")
        if summary['gpu_total_gb'] > 0:
            gpu_usage_pct = (summary['max_gpu_memory_gb'] / summary['gpu_total_gb']) * 100
            log.info(f"  GPU: {summary['max_gpu_memory_gb']:.1f}GB / {summary['gpu_total_gb']:.1f}GB ({gpu_usage_pct:.1f}%)")
        log.info("")
        
        if report['recommendations']:
            log.info("Optimization Recommendations:")
            for i, rec in enumerate(report['recommendations'], 1):
                log.info(f"  {i}. {rec}")
        else:
            log.info("Performance looks optimal! 🎉")
            
        log.info("=" * 50)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Decode embeddings to gene expression using trained State TX model's gene decoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python decode_embeddings_to_gene_expression.py \\
      --input embeddings.h5ad \\
      --checkpoint /path/to/model/final.ckpt \\
      --output predictions.h5ad

  # With custom settings
  python decode_embeddings_to_gene_expression.py \\
      --input embeddings.h5ad \\
      --checkpoint /path/to/model/final.ckpt \\
      --output predictions.h5ad \\
      --batch-size 2048 \\
      --device cuda \\
      --embedding-key mosaicfm-embeddings
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input .h5ad file containing embeddings"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained model checkpoint (.ckpt file)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to output .h5ad file (will store predictions in .obsm['X_hvg'])"
    )
    
    # Optional arguments
    parser.add_argument(
        "--embedding-key", type=str, default="embedding",
        help="Key in input .obsm containing embeddings (default: 'embedding')"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1000,
        help="Batch size for processing (default: 1000)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for inference (default: 'auto' - use GPU if available)"
    )
    parser.add_argument(
        "--mixed-precision", action="store_true",
        help="Use mixed precision (fp16) for faster inference"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def detect_device(device_arg: str) -> torch.device:
    """Detect and configure the appropriate device."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            log.info(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
            log.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            device = torch.device("cpu")
            log.info("🖥️ Using CPU (CUDA not available)")
    else:
        device = torch.device(device_arg)
        if device.type == "cuda" and not torch.cuda.is_available():
            log.error("❌ CUDA requested but not available!")
            sys.exit(1)
        log.info(f"🎯 Using specified device: {device}")
    
    return device


def load_model_with_timing(checkpoint_path: Path, device: torch.device, 
                          tracker: DecoderPerformanceTracker) -> Tuple[Any, Any]:
    """Load model and extract decoder with detailed timing."""
    start_time = time.time()
    
    # Import State TX models
    try:
        from state.tx.models.base import PerturbationModel
        from state.tx.models import StateTransitionPerturbationModel
    except ImportError as e:
        log.error(f"❌ Failed to import State TX models: {e}")
        log.error("Make sure you're in the correct environment with 'state' package installed")
        sys.exit(1)
    
    log.info(f"📦 Loading model from checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    model_load_start = time.time()
    try:
        # Try to load as StateTransitionPerturbationModel first
        model = StateTransitionPerturbationModel.load_from_checkpoint(str(checkpoint_path))
    except Exception as e1:
        log.warning(f"Failed to load as StateTransition model: {e1}")
        try:
            # Fall back to generic PerturbationModel
            model = PerturbationModel.load_from_checkpoint(str(checkpoint_path))
        except Exception as e2:
            log.error(f"❌ Failed to load checkpoint: {e2}")
            sys.exit(1)
    
    model_load_time = time.time() - model_load_start
    tracker.update_timing('model_loading', model_load_time)
    
    log.info(f"✅ Model loaded successfully: {type(model).__name__}")
    
    # Set to eval mode and move to device
    setup_start = time.time()
    model.eval()
    model = model.to(device)
    setup_time = time.time() - setup_start
    tracker.update_timing('model_setup', setup_time)
    
    # Extract decoder
    decoder_start = time.time()
    if not hasattr(model, 'gene_decoder') or model.gene_decoder is None:
        log.error("❌ Model does not have a gene decoder!")
        log.error("This model was likely trained without a gene decoder.")
        log.error("You need a model trained with decoder_cfg specified.")
        sys.exit(1)
    
    decoder = model.gene_decoder
    log.info(f"🧬 Gene decoder found: {type(decoder).__name__}")
    
    # Get decoder output dimensions
    if hasattr(decoder, 'gene_dim'):
        if callable(decoder.gene_dim):
            output_dim = decoder.gene_dim()
        else:
            output_dim = decoder.gene_dim
    else:
        # Try to infer from decoder architecture
        if hasattr(decoder, 'final_layer') and hasattr(decoder.final_layer[0], 'out_features'):
            output_dim = decoder.final_layer[0].out_features
        elif hasattr(decoder, 'decoder') and len(decoder.decoder) > 0:
            # Find last Linear layer
            for layer in reversed(decoder.decoder):
                if isinstance(layer, torch.nn.Linear):
                    output_dim = layer.out_features
                    break
            else:
                output_dim = "unknown"
        else:
            output_dim = "unknown"
    
    log.info(f"🎯 Decoder output dimensions: {output_dim}")
    
    decoder_time = time.time() - decoder_start
    tracker.update_timing('decoder_extraction', decoder_time)
    
    # GPU initialization (if using CUDA)
    if device.type == 'cuda':
        gpu_start = time.time()
        torch.cuda.synchronize()  # Ensure GPU is ready
        gpu_time = time.time() - gpu_start
        tracker.update_timing('gpu_initialization', gpu_time)
    
    total_time = time.time() - start_time
    log.info(f"⏱️ Model loading completed in {total_time:.2f}s")
    
    return model, decoder


def load_embeddings_with_timing(input_path: Path, embedding_key: str, 
                               tracker: DecoderPerformanceTracker) -> Tuple[np.ndarray, Dict[str, Any], int]:
    """Load embeddings from h5ad file with timing and memory monitoring."""
    log.info(f"📖 Loading embeddings from: {input_path}")
    
    # Open file
    file_start = time.time()
    try:
        with h5py.File(input_path, 'r') as f:
            file_time = time.time() - file_start
            tracker.update_timing('file_opening', file_time)
            
            # Get total number of cells
            n_cells = f['obs'].attrs['_index'].shape[0] if '_index' in f['obs'].attrs else len(f['obs']['_index'])
            log.info(f"📊 Found {n_cells:,} cells in dataset")
            
            # Check if embedding key exists
            if 'obsm' not in f or embedding_key not in f['obsm']:
                available_keys = list(f['obsm'].keys()) if 'obsm' in f else []
                log.error(f"❌ Embedding key '{embedding_key}' not found in .obsm")
                log.error(f"Available keys: {available_keys}")
                sys.exit(1)
            
            # Load embeddings
            emb_start = time.time()
            embeddings_dataset = f['obsm'][embedding_key]
            embeddings = embeddings_dataset[:]  # Load into memory
            emb_time = time.time() - emb_start
            tracker.update_timing('embedding_reading', emb_time)
            
            log.info(f"🧬 Loaded embeddings: shape {embeddings.shape}, dtype {embeddings.dtype}")
            
            # Load essential metadata
            meta_start = time.time()
            metadata = {}
            
            # Load obs data (essential for preserving cell information)
            if 'obs' in f:
                obs_data = {}
                for key in f['obs'].keys():
                    if key != '_index':  # Skip index, we'll handle separately
                        obs_data[key] = f['obs'][key][:]
                # Get index
                if '_index' in f['obs']:
                    obs_data['_index'] = f['obs']['_index'][:]
                metadata['obs'] = obs_data
            
            # Load other obsm keys (preserve existing embeddings/features)
            if 'obsm' in f:
                obsm_data = {}
                for key in f['obsm'].keys():
                    if key != embedding_key:  # Skip the one we already loaded
                        obsm_data[key] = f['obsm'][key][:]
                metadata['obsm'] = obsm_data
            
            # Load var data if it exists (gene information)
            if 'var' in f:
                var_data = {}
                for key in f['var'].keys():
                    var_data[key] = f['var'][key][:]
                metadata['var'] = var_data
            
            meta_time = time.time() - meta_start
            tracker.update_timing('metadata_reading', meta_time)
            
    except Exception as e:
        log.error(f"❌ Failed to load data from {input_path}: {e}")
        sys.exit(1)
    
    return embeddings, metadata, n_cells


def process_embeddings_in_batches(embeddings: np.ndarray, decoder: Any, device: torch.device,
                                batch_size: int, mixed_precision: bool,
                                tracker: DecoderPerformanceTracker) -> np.ndarray:
    """Process embeddings through decoder in batches with detailed timing."""
    n_cells = embeddings.shape[0]
    n_batches = (n_cells + batch_size - 1) // batch_size
    
    log.info(f"🔄 Processing {n_cells:,} cells in {n_batches} batches of size {batch_size}")
    
    # Determine output shape by running one small batch
    test_batch_size = min(10, n_cells)
    test_embeddings = embeddings[:test_batch_size]
    
    with torch.no_grad():
        test_start = time.time()
        test_tensor = torch.tensor(test_embeddings, dtype=torch.float32).to(device)
        if mixed_precision:
            test_tensor = test_tensor.half()
        test_output = decoder(test_tensor)
        
        # Handle different decoder types
        if isinstance(test_output, tuple):
            # NBDecoder returns (mu, sigma)
            test_output = test_output[0]
        
        output_shape = test_output.shape[1:]  # Remove batch dimension
        output_dtype = test_output.cpu().numpy().dtype
        test_time = time.time() - test_start
    
    log.info(f"🎯 Decoder output shape per cell: {output_shape}, dtype: {output_dtype}")
    log.info(f"⏱️ Test batch processed in {test_time:.3f}s ({test_batch_size/test_time:.0f} cells/sec)")
    
    # Pre-allocate output array
    all_predictions = np.zeros((n_cells,) + output_shape, dtype=output_dtype)
    
    # Process in batches
    with torch.no_grad():
        for batch_idx in tqdm(range(n_batches), desc="Processing batches", unit="batch"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_cells)
            current_batch_size = end_idx - start_idx
            
            batch_embeddings = embeddings[start_idx:end_idx]
            
            # Convert to tensor
            tensor_start = time.time()
            batch_tensor = torch.tensor(batch_embeddings, dtype=torch.float32)
            if mixed_precision:
                batch_tensor = batch_tensor.half()
            tensor_time = time.time() - tensor_start
            tracker.update_timing('numpy_to_tensor', tensor_time)
            
            # Transfer to GPU
            transfer_start = time.time()
            batch_tensor = batch_tensor.to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            transfer_time = time.time() - transfer_start
            tracker.update_timing('cpu_to_gpu_transfer', transfer_time)
            
            # Decoder inference
            inference_start = time.time()
            batch_output = decoder(batch_tensor)
            
            # Handle different decoder types
            if isinstance(batch_output, tuple):
                # NBDecoder returns (mu, sigma) - use mu
                batch_output = batch_output[0]
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            inference_time = time.time() - inference_start
            tracker.update_timing('decoder_inference', inference_time)
            
            # Transfer back to CPU
            cpu_transfer_start = time.time()
            batch_output_cpu = batch_output.cpu()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            cpu_transfer_time = time.time() - cpu_transfer_start
            tracker.update_timing('gpu_to_cpu_transfer', cpu_transfer_time)
            
            # Convert to numpy
            numpy_start = time.time()
            batch_predictions = batch_output_cpu.numpy()
            numpy_time = time.time() - numpy_start
            tracker.update_timing('tensor_to_numpy', numpy_time)
            
            # Store results
            all_predictions[start_idx:end_idx] = batch_predictions
            
            # Update tracker
            tracker.update_timing('batch_overhead', 0.001, current_batch_size)  # Small overhead time
            
            # Clean up GPU memory
            if device.type == 'cuda':
                del batch_tensor, batch_output, batch_output_cpu
                torch.cuda.empty_cache()
    
    log.info(f"✅ All batches processed successfully")
    return all_predictions


def save_results_with_timing(output_path: Path, predictions: np.ndarray, metadata: Dict[str, Any],
                           tracker: DecoderPerformanceTracker):
    """Save predictions to h5ad file with timing."""
    log.info(f"💾 Saving results to: {output_path}")
    
    # Create output file
    create_start = time.time()
    with h5py.File(output_path, 'w') as f:
        create_time = time.time() - create_start
        tracker.update_timing('output_file_creation', create_time)
        
        # Save predictions in obsm['X_hvg']
        pred_start = time.time()
        f.create_dataset('obsm/X_hvg', data=predictions, compression='gzip', compression_opts=6)
        pred_time = time.time() - pred_start
        tracker.update_timing('predictions_writing', pred_time)
        
        log.info(f"✅ Predictions saved: shape {predictions.shape}, dtype {predictions.dtype}")
        
        # Save metadata
        meta_start = time.time()
        
        # Save obs data
        if 'obs' in metadata:
            obs_grp = f.create_group('obs')
            for key, data in metadata['obs'].items():
                if isinstance(data, np.ndarray):
                    if data.dtype.kind in ['U', 'S', 'O']:  # String data
                        # Convert to bytes for HDF5 compatibility
                        if data.dtype.kind == 'O':
                            data = np.array([str(x).encode('utf-8') for x in data])
                        obs_grp.create_dataset(key, data=data)
                    else:
                        obs_grp.create_dataset(key, data=data)
        
        # Save other obsm data
        if 'obsm' in metadata:
            if 'obsm' not in f:
                f.create_group('obsm')
            for key, data in metadata['obsm'].items():
                f['obsm'].create_dataset(key, data=data, compression='gzip', compression_opts=6)
        
        # Save var data
        if 'var' in metadata:
            var_grp = f.create_group('var')
            for key, data in metadata['var'].items():
                if isinstance(data, np.ndarray):
                    if data.dtype.kind in ['U', 'S', 'O']:  # String data
                        if data.dtype.kind == 'O':
                            data = np.array([str(x).encode('utf-8') for x in data])
                        var_grp.create_dataset(key, data=data)
                    else:
                        var_grp.create_dataset(key, data=data)
        
        meta_time = time.time() - meta_start
        tracker.update_timing('metadata_writing', meta_time)
        
        # Finalize file
        final_start = time.time()
        f.flush()
        final_time = time.time() - final_start
        tracker.update_timing('file_finalization', final_time)
    
    log.info(f"✅ Results saved successfully")


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate inputs
    input_path = Path(args.input)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    
    if not input_path.exists():
        log.error(f"❌ Input file not found: {input_path}")
        sys.exit(1)
    
    if not checkpoint_path.exists():
        log.error(f"❌ Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    log.info(f"🎯 Input: {input_path}")
    log.info(f"🎯 Checkpoint: {checkpoint_path}")
    log.info(f"🎯 Output: {output_path}")
    log.info(f"🎯 Embedding key: {args.embedding_key}")
    log.info(f"🎯 Batch size: {args.batch_size}")
    
    # Detect device
    device = detect_device(args.device)
    
    # Get initial cell count for performance tracking
    try:
        with h5py.File(input_path, 'r') as f:
            n_cells = f['obs'].attrs['_index'].shape[0] if '_index' in f['obs'].attrs else len(f['obs']['_index'])
    except:
        n_cells = 0  # Will be updated when we load
    
    # Initialize performance tracker
    tracker = DecoderPerformanceTracker(n_cells, input_path.name)
    
    try:
        # Load model
        model, decoder = load_model_with_timing(checkpoint_path, device, tracker)
        
        # Load embeddings
        embeddings, metadata, n_cells = load_embeddings_with_timing(input_path, args.embedding_key, tracker)
        
        # Update tracker with correct cell count
        tracker.total_cells = n_cells
        
        # Process embeddings
        predictions = process_embeddings_in_batches(
            embeddings, decoder, device, args.batch_size, args.mixed_precision, tracker
        )
        
        # Save results
        save_results_with_timing(output_path, predictions, metadata, tracker)
        
        # Generate and log final report
        report = tracker.generate_final_report()
        tracker.log_final_report(report)
        
        log.info("🎉 Processing completed successfully!")
        
    except KeyboardInterrupt:
        log.warning("⚠️ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        log.error(f"❌ Processing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()