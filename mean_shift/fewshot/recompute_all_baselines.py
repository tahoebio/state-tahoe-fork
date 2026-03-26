#!/usr/bin/env python
# coding: utf-8

# # Recompute All Mean Shift Baselines for Few-Shot (CORRECTED + PER-PLATE)
# 
# This notebook computes all mean shift baselines for few-shot generalization with corrected algorithms and per-plate processing.
# 
# ## Algorithm Fix Applied:
# 
# **Corrected Mean Shift Computation**: Compute shifts WITHIN training cell lines (not mixing embeddings from different sources).
# 
# **Per-Plate Processing**: All operations respect plate boundaries to avoid batch effects:
# - Control mapping: per (cell_line, plate)
# - Mean shift ablation: per (cell_line, drug, plate)
# - Nearest-cell-line: per (drug, plate)
# - Cross-cell-line: per (drug, plate) with two versions
# 
# ## Scale Note:
# 
# Few-shot data does NOT need scale correction because:
# - Training centroids: norm ~47.5
# - Test data: norm ~58.3
# - Both are unnormalized and at similar scale (ratio ~0.8)
# 
# This is different from zero-shot where test data is normalized (norm ~1).
# 
# ## Baselines Computed:
# 
# 1. **Nearest-Cell-Line Mean Shift**: Find closest training cell line by DMSO similarity (per plate), use its shift
# 2. **Cross-Cell-Line V1 (DMSO-corrected)**: Average shifts computed within each training cell line (per plate)
# 3. **Cross-Cell-Line V2 (Raw mean)**: Use mean of training drug embeddings minus test DMSO (per plate)
# 4. **Mean Shift Ablation**: Oracle baseline using test data (per cell line, drug, plate)
# 5. **Control Passthrough**: Do-nothing baseline (use control embedding, no shift)
# 
# ## Key Difference from Zero-Shot:
# 
# In few-shot, the test cell lines ARE in the training set (only specific drugs are held out). So:
# - **Test DMSO centroids come from training centroids** (not computed from test data)
# - Training centroids include DMSO for test cell lines
# - We filter out only the specific test (cell_line, drug) pairs

# In[ ]:


import anndata as ad
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from tqdm.notebook import tqdm
import toml

print("Imports successful!")


# ## Configuration

# In[ ]:


# Paths
TOML_PATH = '/tahoe/generalization_converted_cell_lines_3b.toml'
CENTROIDS_DIR = '/tahoe/centroids'
REAL_PATH = '/tahoe/data/real.h5ad'

# Outputs
OUTPUT_NEAREST = '/tahoe/data/pred_nearest_cell_line_corrected.h5ad'
OUTPUT_CROSS_V1 = '/tahoe/data/pred_cross_cell_line_v1_dmso_corrected.h5ad'
OUTPUT_CROSS_V2 = '/tahoe/data/pred_cross_cell_line_v2_raw_mean.h5ad'
OUTPUT_ABLATION = '/tahoe/data/pred_mean_shift_ablation.h5ad'
OUTPUT_CONTROL = '/tahoe/data/pred_control_passthrough.h5ad'
MAPPING_PATH = '/tahoe/data/control_to_perturbed_mapping_per_plate.pkl'

# Constants
EMBED_KEY = 'mosaicfm-70m-merged'
CONTROL_DRUG = "[('DMSO_TF', 0.0, 'uM')]"

print(f"Configuration loaded")
print(f"  TOML: {TOML_PATH}")
print(f"  Centroids: {CENTROIDS_DIR}")
print(f"  Test data: {REAL_PATH}")


# ## Step 1: Parse TOML for Test Pairs

# In[ ]:


def parse_toml_for_test_pairs(toml_path):
    """
    Parse TOML config to extract test (cell_line, drug) pairs.
    
    Returns:
        set: Set of (cell_line, drug_string) tuples
    """
    config = toml.load(toml_path)
    test_pairs = set()
    
    # Extract from fewshot section
    fewshot = config.get('fewshot', {})
    for key, splits in fewshot.items():
        # key format: "tahoe.CVCL_1097"
        if '.' in key:
            cell_line = key.split('.', 1)[1]
            
            # Get test perturbations
            test_perts = splits.get('test', [])
            for pert_string in test_perts:
                test_pairs.add((cell_line, pert_string))
    
    return test_pairs

test_pairs = parse_toml_for_test_pairs(TOML_PATH)
print(f"Test pairs: {len(test_pairs)}")
print(f"\nFirst 10 test pairs:")
for pair in list(test_pairs)[:10]:
    print(f"  {pair}")


# ## Step 2: Load Training Centroids (No Scale Correction Needed)

# In[ ]:


def load_all_centroids(centroid_dir, embed_key='mosaicfm-70m-merged'):
    """
    Load and combine all centroid files WITH PLATE INFORMATION.
    
    Plate info is extracted from filename: plate_plate10_centroids.h5ad -> plate10
    
    Returns:
        pd.DataFrame: Combined centroids with columns [cell_line_id, drugname_drugconc, plate, embedding]
    """
    centroid_dir = Path(centroid_dir)
    centroid_files = sorted(centroid_dir.glob('plate_plate*_centroids.h5ad'))
    
    print(f"Loading {len(centroid_files)} centroid files...")
    
    all_centroids = []
    for file_path in tqdm(centroid_files, desc="Loading centroids"):
        # Extract plate name from filename: plate_plate10_centroids.h5ad -> plate10
        plate_name = file_path.stem.replace('_centroids', '').replace('plate_', '')
        
        adata = ad.read_h5ad(file_path)
        
        df = pd.DataFrame({
            'cell_line_id': adata.obs['cell_line_id'].values,
            'drugname_drugconc': adata.obs['drugname_drugconc'].values,
            'plate': plate_name,  # Add plate info from filename
        })
        df['embedding'] = list(adata.obsm[embed_key])
        
        all_centroids.append(df)
    
    combined = pd.concat(all_centroids, ignore_index=True)
    print(f"\nLoaded {len(combined)} training centroids")
    print(f"  Cell lines: {combined['cell_line_id'].nunique()}")
    print(f"  Drugs: {combined['drugname_drugconc'].nunique()}")
    print(f"  Plates: {combined['plate'].nunique()}")
    print(f"  Plate names: {sorted(combined['plate'].unique())}")
    
    return combined

all_centroids = load_all_centroids(CENTROIDS_DIR, embed_key=EMBED_KEY)


# In[ ]:


# Filter to TRAIN set only (exclude test pairs, but KEEP all DMSO controls)
print("\nFiltering centroids to TRAIN set...")
print(f"Before filtering: {len(all_centroids)} centroids")

# Create mask for test pairs, but NEVER filter out DMSO controls
is_test = all_centroids.apply(
    lambda row: (
        (row['cell_line_id'], row['drugname_drugconc']) in test_pairs and
        row['drugname_drugconc'] != CONTROL_DRUG  # Never filter DMSO
    ),
    axis=1
)

train_centroids = all_centroids[~is_test].copy()
test_centroids = all_centroids[is_test].copy()

print(f"After filtering:")
print(f"  Train centroids: {len(train_centroids)}")
print(f"  Test centroids (excluded): {len(test_centroids)}")
print(f"\nTrain set stats:")
print(f"  Cell lines: {train_centroids['cell_line_id'].nunique()}")
print(f"  Drugs: {train_centroids['drugname_drugconc'].nunique()}")

# Check DMSO controls are preserved
dmso_controls = train_centroids[train_centroids['drugname_drugconc'] == CONTROL_DRUG]
print(f"\nDMSO controls in train set:")
print(f"  Total DMSO centroids: {len(dmso_controls)}")
print(f"  Cell lines with DMSO: {dmso_controls['cell_line_id'].nunique()}")


# ## Step 3: Filter to Training Set (Exclude Test Pairs)

# In[ ]:


print("Loading test data (backed mode)...")
real_adata = ad.read_h5ad(REAL_PATH, backed='r')

print(f"Test data shape: {real_adata.shape}")

print("\nLoading embeddings into memory...")
embeddings = real_adata.obsm[EMBED_KEY][:]
print(f"Embeddings shape: {embeddings.shape}")

# Count control vs perturbed
is_control = real_adata.obs['drugname_drugconc'] == CONTROL_DRUG
n_control = is_control.sum()
n_perturbed = (~is_control).sum()

print(f"\nCell breakdown:")
print(f"  Control cells (DMSO): {n_control:,}")
print(f"  Perturbed cells: {n_perturbed:,}")
print(f"  Total: {len(real_adata):,}")


# ## Step 4: Load Test Data

# In[ ]:


# Decode plate one-hot encoding to string names
# Mapping from advisor's batch_onehot_map.pkl
onehot_to_plate = {
    tuple([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]): 'plate1',
    tuple([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]): 'plate10',
    tuple([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]): 'plate11',
    tuple([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]): 'plate12',
    tuple([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]): 'plate13',
    tuple([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]): 'plate14',
    tuple([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]): 'plate2',
    tuple([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]): 'plate3',
    tuple([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]): 'plate4',
    tuple([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]): 'plate5',
    tuple([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]): 'plate6',
    tuple([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]): 'plate7',
    tuple([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]): 'plate8',
    tuple([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]): 'plate9',
}

def decode_plate_onehot(plate_value):
    """Decode one-hot plate vector to string name"""
    # If already a decoded string, return it
    if isinstance(plate_value, str):
        if plate_value.startswith('plate'):
            return plate_value
        
        # Parse string representation like "[0. 0. 1. 0. ...]"
        if plate_value.startswith('[') and plate_value.endswith(']'):
            try:
                # Remove brackets and split by whitespace
                inner = plate_value.strip('[]')
                # Split and convert to floats
                values = [float(x.strip()) for x in inner.split() if x.strip()]
                
                if len(values) == 14:
                    plate_tuple = tuple(values)
                    decoded = onehot_to_plate.get(plate_tuple, None)
                    if decoded is not None:
                        return decoded
                    
                    # Try rounding in case of floating point errors
                    plate_tuple_rounded = tuple([round(v) for v in values])
                    decoded = onehot_to_plate.get(plate_tuple_rounded, None)
                    if decoded is not None:
                        return decoded
                    
                    print(f"⚠️  Warning: Unknown plate encoding from string: {values}")
                    return 'unknown'
            except Exception as e:
                print(f"⚠️  Error parsing plate string: {e}, value: {plate_value}")
                return 'unknown'
    
    # Try as array/list
    try:
        import numpy as np
        plate_array = np.array(plate_value, dtype=float)
        
        if plate_array.shape == (14,):
            plate_tuple = tuple(plate_array.tolist())
            decoded = onehot_to_plate.get(plate_tuple, None)
            
            if decoded is not None:
                return decoded
            
            # Try rounding
            plate_tuple_rounded = tuple([round(float(x)) for x in plate_array])
            decoded = onehot_to_plate.get(plate_tuple_rounded, None)
            if decoded is not None:
                return decoded
            
            print(f"⚠️  Warning: Unknown plate encoding: {plate_array}")
            return 'unknown'
    except Exception as e:
        print(f"⚠️  Error decoding plate: {e}, value: {plate_value}")
        return 'unknown'
    
    return 'unknown'

print("Testing plate decoder...")
# Test with a sample plate value
test_idx = np.random.choice(len(real_adata))
test_plate = real_adata.obs['plate'].iloc[test_idx]
decoded = decode_plate_onehot(test_plate)
print(f"  Sample plate encoding: {test_plate}")
print(f"  Decoded to: {decoded}")
assert decoded.startswith('plate'), f"Decoder not working! Got: {decoded}"
print("✓ Plate decoding function ready")


# In[ ]:


# Load existing control mapping (already generated)
if Path(MAPPING_PATH).exists():
    print(f"Loading existing mapping from {MAPPING_PATH}...")
    with open(MAPPING_PATH, 'rb') as f:
        control_mapping = pickle.load(f)
    print(f"✓ Loaded mapping for {len(control_mapping):,} perturbed cells")
else:
    raise FileNotFoundError(
        f"Control mapping not found at {MAPPING_PATH}. "
        f"Please run the create_control_mapping() function first."
    )


# ## Step 5: Create Deterministic Control-to-Perturbed Mapping

# In[ ]:


def compute_mean_shifts(test_pairs, train_centroids, control_drug):
    """
    Compute nearest-cell-line and both cross-cell-line mean shifts PER PLATE.
    
    KEY DIFFERENCE FROM ZERO-SHOT:
    In few-shot, test DMSO centroids come from training centroids (not computed from test data).
    
    NEAREST (per plate):
    1. Get test cell line's DMSO centroid from training centroids (for this plate)
    2. Find closest training cell line by DMSO similarity (same plate)
    3. Use that cell line's shift: shift = drug_closest - DMSO_closest
    
    CROSS V1 - DMSO-corrected (per plate):
    1. Get test cell line's DMSO centroid from training centroids (for this plate)
    2. Compute shift for each training cell line (same plate): shift_i = drug_i - DMSO_i
    3. Average all shifts
    
    CROSS V2 - Raw mean (per plate):
    1. Get test cell line's DMSO centroid from training centroids (for this plate)
    2. Compute mean drug embedding across training cell lines (same plate): mean_drug = mean(drug_i)
    3. Compute shift: shift = mean_drug - test_DMSO
    """
    nearest_shifts = {}
    cross_v1_shifts = {}
    cross_v2_shifts = {}
    
    print("Computing mean shifts (per plate)...")
    
    skipped_no_dmso = 0
    skipped_no_candidates = 0
    
    # Expand test_pairs to include plate information
    # We'll iterate over (cell_line, drug, plate) combinations found in training data
    test_combos = set()
    for test_cell_line, test_drug in test_pairs:
        # Find all plates where this (cell_line, drug) exists in test set
        relevant_plates = train_centroids[
            (train_centroids['cell_line_id'] == test_cell_line) &
            (train_centroids['drugname_drugconc'] == control_drug)
        ]['plate'].unique()
        
        for plate in relevant_plates:
            test_combos.add((test_cell_line, test_drug, plate))
    
    print(f"Processing {len(test_combos)} (cell_line, drug, plate) combinations...")
    
    for test_cell_line, test_drug, test_plate in tqdm(test_combos, desc="Computing shifts"):
        # Step 1: Get test cell line's DMSO centroid from TRAINING centroids (for this plate)
        dmso_mask = (train_centroids['cell_line_id'] == test_cell_line) & \
                   (train_centroids['drugname_drugconc'] == control_drug) & \
                   (train_centroids['plate'] == test_plate)
        dmso_cents = train_centroids[dmso_mask]
        
        if len(dmso_cents) == 0:
            skipped_no_dmso += 1
            continue
        
        # Average if multiple centroids (shouldn't happen per-plate, but just in case)
        dmso_test = np.mean(np.stack(dmso_cents['embedding'].values), axis=0)
        
        # Step 2: Find all training cell lines that have BOTH DMSO and this drug ON THIS PLATE
        train_drug_mask = (train_centroids['drugname_drugconc'] == test_drug) & \
                         (train_centroids['plate'] == test_plate)
        train_with_drug = train_centroids[train_drug_mask]['cell_line_id'].unique()
        
        train_dmso_mask = (train_centroids['drugname_drugconc'] == control_drug) & \
                         (train_centroids['plate'] == test_plate)
        train_with_dmso = train_centroids[train_dmso_mask]['cell_line_id'].unique()
        
        # Exclude test cell line itself
        candidate_cell_lines = (set(train_with_drug) & set(train_with_dmso)) - {test_cell_line}
        
        if len(candidate_cell_lines) == 0:
            skipped_no_candidates += 1
            continue
        
        # Step 3: For each candidate, compute shift within that training cell line
        all_shifts = []  # For Cross V1
        all_drug_embeddings = []  # For Cross V2
        min_distance = float('inf')
        nearest_shift = None
        
        for train_cell_line in candidate_cell_lines:
            # Get training DMSO centroid (this plate)
            train_dmso_mask = (train_centroids['cell_line_id'] == train_cell_line) & \
                             (train_centroids['drugname_drugconc'] == control_drug) & \
                             (train_centroids['plate'] == test_plate)
            train_dmso_cents = train_centroids[train_dmso_mask]
            
            if len(train_dmso_cents) == 0:
                continue
            
            train_dmso = np.mean(np.stack(train_dmso_cents['embedding'].values), axis=0)
            
            # Get training drug centroid (this plate)
            train_drug_mask = (train_centroids['cell_line_id'] == train_cell_line) & \
                             (train_centroids['drugname_drugconc'] == test_drug) & \
                             (train_centroids['plate'] == test_plate)
            train_drug_cents = train_centroids[train_drug_mask]
            
            if len(train_drug_cents) == 0:
                continue
            
            train_drug = np.mean(np.stack(train_drug_cents['embedding'].values), axis=0)
            
            # Collect drug embedding for Cross V2
            all_drug_embeddings.append(train_drug)
            
            # Compute shift within this training cell line for Cross V1
            shift = train_drug - train_dmso
            all_shifts.append(shift)
            
            # For NEAREST: track which has closest DMSO
            distance = np.linalg.norm(dmso_test - train_dmso)
            if distance < min_distance:
                min_distance = distance
                nearest_shift = shift
        
        # Store shifts
        key = (test_cell_line, test_drug, test_plate)
        
        if nearest_shift is not None:
            nearest_shifts[key] = nearest_shift
        
        if len(all_shifts) > 0:
            # Cross V1: Average of shifts (DMSO-corrected)
            cross_v1_shifts[key] = np.mean(all_shifts, axis=0)
        
        if len(all_drug_embeddings) > 0:
            # Cross V2: Mean drug embedding - test DMSO (raw mean)
            mean_drug = np.mean(all_drug_embeddings, axis=0)
            cross_v2_shifts[key] = mean_drug - dmso_test
    
    print(f"\nResults:")
    print(f"  Nearest shifts computed: {len(nearest_shifts)}")
    print(f"  Cross V1 (DMSO-corrected) shifts computed: {len(cross_v1_shifts)}")
    print(f"  Cross V2 (raw mean) shifts computed: {len(cross_v2_shifts)}")
    print(f"  Skipped (no test DMSO): {skipped_no_dmso}")
    print(f"  Skipped (no training candidates): {skipped_no_candidates}")
    
    return nearest_shifts, cross_v1_shifts, cross_v2_shifts

nearest_shifts, cross_v1_shifts, cross_v2_shifts = compute_mean_shifts(test_pairs, train_centroids, CONTROL_DRUG)


# ## Step 6: Compute Mean Shifts (Nearest & Cross)

# In[ ]:


def compute_ablation_shifts(real_adata, embeddings, control_drug):
    """
    Compute mean shift ablation PER PLATE: use test data to compute shifts (oracle).
    
    For each (cell_line, drug, plate) in test data:
    - Compute mean drug embedding
    - Compute mean DMSO embedding for that (cell_line, plate)
    - shift = mean_drug - mean_DMSO
    """
    ablation_shifts = {}
    
    print("\nComputing mean shift ablation (oracle, per plate)...")
    
    # OPTIMIZATION: Decode all plates once upfront
    print("  Decoding all plate values...")
    decoded_plates = [decode_plate_onehot(p) for p in real_adata.obs['plate']]
    
    # Create a DataFrame for vectorized operations
    print("  Creating lookup DataFrame...")
    df = pd.DataFrame({
        'cell_line': real_adata.obs['cell_line_id'].values,
        'drug': real_adata.obs['drugname_drugconc'].values,
        'plate': decoded_plates,
        'idx': np.arange(len(real_adata))
    })
    
    # Get unique (cell_line, drug, plate) triplets (exclude DMSO)
    print("  Finding unique triplets...")
    triplets_df = df[df['drug'] != control_drug][['cell_line', 'drug', 'plate']].drop_duplicates()
    
    print(f"  Processing {len(triplets_df)} triplets...")
    
    # Group by (cell_line, drug, plate) for drug embeddings
    drug_groups = df[df['drug'] != control_drug].groupby(['cell_line', 'drug', 'plate'])['idx'].apply(list).to_dict()
    
    # Group by (cell_line, plate) for DMSO embeddings
    dmso_groups = df[df['drug'] == control_drug].groupby(['cell_line', 'plate'])['idx'].apply(list).to_dict()
    
    print("  Computing shifts...")
    for _, row in tqdm(triplets_df.iterrows(), total=len(triplets_df), desc="Computing ablation"):
        cell_line = row['cell_line']
        drug = row['drug']
        plate = row['plate']
        
        # Get drug indices from precomputed groups
        drug_key = (cell_line, drug, plate)
        if drug_key not in drug_groups:
            continue
        drug_indices = drug_groups[drug_key]
        
        # Get DMSO indices from precomputed groups
        dmso_key = (cell_line, plate)
        if dmso_key not in dmso_groups:
            continue
        dmso_indices = dmso_groups[dmso_key]
        
        # Compute means
        drug_mean = np.mean(embeddings[drug_indices], axis=0)
        dmso_mean = np.mean(embeddings[dmso_indices], axis=0)
        
        # Store shift
        shift = drug_mean - dmso_mean
        ablation_shifts[(cell_line, drug, plate)] = shift
    
    print(f"Computed {len(ablation_shifts)} ablation shifts")
    
    return ablation_shifts

ablation_shifts = compute_ablation_shifts(real_adata, embeddings, CONTROL_DRUG)


# ## Step 7: Compute Mean Shift Ablation (Oracle)
# 
# This baseline uses test data to compute mean shifts (oracle/upper bound).

# In[ ]:


def apply_shifts(real_adata, embeddings, control_mapping, shift_table, control_drug, name):
    """
    Apply mean shifts to test cells (with plate information in keys).
    """
    n_cells = len(real_adata)
    predictions = np.zeros((n_cells, embeddings.shape[1]), dtype=np.float32)
    
    print(f"\nApplying {name} shifts to {n_cells:,} cells...")
    
    # OPTIMIZATION: Decode all plates once upfront
    print("  Decoding all plate values...")
    decoded_plates = np.array([decode_plate_onehot(p) for p in real_adata.obs['plate']])
    
    # Pre-fetch arrays for faster access
    cell_lines = real_adata.obs['cell_line_id'].values
    drugs = real_adata.obs['drugname_drugconc'].values
    
    # Step 1: Handle control cells (vectorized)
    print("  Handling control cells...")
    control_mask = drugs == control_drug
    predictions[control_mask] = embeddings[control_mask]
    n_control = control_mask.sum()
    
    # Step 2: Handle perturbed cells (need loop for shift lookup)
    print("  Applying shifts to perturbed cells...")
    perturbed_indices = np.where(~control_mask)[0]
    
    n_shifted = 0
    n_fallback = 0
    
    for i in tqdm(perturbed_indices, desc=f"Applying {name}"):
        # Check if control mapping exists
        if i not in control_mapping:
            predictions[i] = embeddings[i]
            n_fallback += 1
            continue
        
        control_idx = control_mapping[i]
        control_emb = embeddings[control_idx]
        
        # Look up shift
        key = (cell_lines[i], drugs[i], decoded_plates[i])
        if key not in shift_table:
            predictions[i] = embeddings[i]
            n_fallback += 1
            continue
        
        predictions[i] = control_emb + shift_table[key]
        n_shifted += 1
    
    print(f"Results:")
    print(f"  Control: {n_control:,}")
    print(f"  Shifted: {n_shifted:,}")
    print(f"  Fallback: {n_fallback:,}")
    
    return predictions

print("\n" + "="*80)
print("APPLYING SHIFTS")
print("="*80)

predictions_nearest = apply_shifts(
    real_adata, embeddings, control_mapping, nearest_shifts, CONTROL_DRUG, "Nearest"
)

predictions_cross_v1 = apply_shifts(
    real_adata, embeddings, control_mapping, cross_v1_shifts, CONTROL_DRUG, "Cross V1"
)

predictions_cross_v2 = apply_shifts(
    real_adata, embeddings, control_mapping, cross_v2_shifts, CONTROL_DRUG, "Cross V2"
)

predictions_ablation = apply_shifts(
    real_adata, embeddings, control_mapping, ablation_shifts, CONTROL_DRUG, "Ablation"
)


# ## Step 8: Apply Shifts and Generate Predictions

# In[ ]:


def create_control_passthrough(real_adata, embeddings, control_mapping, control_drug):
    """
    Create control passthrough baseline (no shift applied).
    """
    n_cells = len(real_adata)
    predictions = np.zeros((n_cells, embeddings.shape[1]), dtype=np.float32)
    
    print(f"\nCreating control passthrough baseline...")
    
    for i in tqdm(range(n_cells), desc="Control passthrough"):
        drug = real_adata.obs['drugname_drugconc'].iloc[i]
        
        if drug == control_drug:
            predictions[i] = embeddings[i]
        else:
            if i in control_mapping:
                control_idx = control_mapping[i]
                predictions[i] = embeddings[control_idx]
            else:
                predictions[i] = embeddings[i]
    
    return predictions

predictions_control = create_control_passthrough(
    real_adata, embeddings, control_mapping, CONTROL_DRUG
)


# ## Step 9: Generate Control Passthrough Baseline

# In[ ]:


def save_predictions(real_adata, predictions, output_path, embed_key):
    pred_adata = ad.AnnData(
        X=real_adata.X,
        obs=real_adata.obs.copy()
    )
    pred_adata.obsm[embed_key] = predictions
    pred_adata.write_h5ad(output_path)
    print(f"  Saved: {output_path}")

print("\n" + "="*80)
print("SAVING PREDICTIONS")
print("="*80 + "\n")

save_predictions(real_adata, predictions_nearest, OUTPUT_NEAREST, EMBED_KEY)
save_predictions(real_adata, predictions_cross_v1, OUTPUT_CROSS_V1, EMBED_KEY)
save_predictions(real_adata, predictions_cross_v2, OUTPUT_CROSS_V2, EMBED_KEY)
save_predictions(real_adata, predictions_ablation, OUTPUT_ABLATION, EMBED_KEY)
save_predictions(real_adata, predictions_control, OUTPUT_CONTROL, EMBED_KEY)

print("\n" + "="*80)
print("ALL BASELINES COMPLETE")
print("="*80)
print(f"\n1. Nearest-Cell-Line: {OUTPUT_NEAREST}")
print(f"2. Cross-Cell-Line V1 (DMSO-corrected): {OUTPUT_CROSS_V1}")
print(f"3. Cross-Cell-Line V2 (raw mean): {OUTPUT_CROSS_V2}")
print(f"4. Mean Shift Ablation: {OUTPUT_ABLATION}")
print(f"5. Control Passthrough: {OUTPUT_CONTROL}")
print(f"\nNext step: Run MMD evaluation on all baselines")

