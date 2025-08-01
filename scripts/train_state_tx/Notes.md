# State training notes 250708
- WandB logging error: Warning: Failed to initialize wandb logger: Error uploading run: returned error 404: {"data":{"upsertBucket":null},"errors":[{"message":"entity your_entity_name not found during upsertBucket","path":["upsertBucket"]}]}                                                 Continuing without wandb logging
- Training took 2:29
- wandb: 🚀 View run tahoe_state_tx_20250707_231229 at: https://wandb.ai/your_entity_name/state_tx_tahoe/runs/jwof5xow
- It seems there is a setting for WandB entity somewhere
- Model saved in: /tahoe/drive_3/ANALYSIS/analysis_190/Code/train_state_tx/experiments/tahoe_state_tx_20250707_231229
- Running evaluation failed. Probably because input file has 0 cells for eval cases (because we split on the dataset level, not with just metadata)
- Log reports 992,747 train cells and 7,253 test cells. This is weird: there should be 0 test cells.
- Are the 7,253 cells DMSO cells that _could_ be transported?
- No, there are ~24,000 DMSO cells in the 1M subsample example data.
- Prediction only predicts ~7,253 cells? 


python scripts/dataset2anndata.py /tahoe/drive_3/ANALYSIS/analysis_190/Data/20250618.tahoe_embeddings_70M /tahoe/drive_3/ANALYSIS/analysis_190/Data/state_input/20250618.tahoe_embeddings_70M.h5ad --verbose


state tx infer --adata /tahoe/drive_3/ANALYSIS/analysis_190/Data/20250618.tahoe_embeddings_70M.train.1M.sample.CONTROL.h5ad --embed_key mosaicfm-70m-merged --pert_col drug_dose --output /tahoe/drive_3/ANALYSIS/analysis_190/Code/train_state_tx/experiments/tahoe_state_tx_20250707_231229/20250618.tahoe_embeddings_70M.train.1M.sample.CONTROL.with_pred.h5ad --model_dir /tahoe/drive_3/ANALYSIS/analysis_190/Code/train_state_tx/experiments/tahoe_state_tx_20250707_231229 --celltype_col cell_line --celltypes CVCL_0179,CVCL_0320,CVCL_1094,CVCL_1097,CVCL_1119,CVCL_1239,CVCL_1495,CVCL_1693,CVCL_1724,CVCL_1731

# State TX Inference on Held-Out Perturbations - 2025-07-09

## Background
- Created expanded dataset from dmso_controls.h5ad (15,000 DMSO control cells)
- Mapped to 228 held-out perturbations from fewshot DC split
- Total inference dataset: 3,420,000 samples (15,000 cells × 228 perturbations)

## Command Running in Background
```bash
tmux new-session -d -s state_inference 'state tx infer --adata dmso_controls_expanded_for_inference.h5ad --model_dir train_state_tx/experiments/tahoe_state_tx_20250707_231229 --embed_key embedding --pert_col drug_dose --output dmso_controls_predicted_all_perturbations.h5ad 2>&1 | tee state_inference.log'
```

## Monitoring
- **Session**: `tmux attach-session -t state_inference` (Ctrl+b, d to detach)
- **Log**: `tail -f state_inference.log`
- **Check status**: `tmux list-sessions`

## Expected Performance
- **Processing speed**: ~4,500 samples/second
- **Estimated time**: 12-15 hours for 3.4M samples
- **Output**: `dmso_controls_predicted_all_perturbations.h5ad`

## Output Structure
- **Shape**: 3,420,000 × 512 (cells × genes)
- **Predictions**: `obsm['model_preds']` - predicted perturbed embeddings
- **Metadata**: Tracks original cell index and perturbation for each prediction
- **Use case**: Analyze how control cells respond to held-out test perturbations

## Held-Out Perturbations (228 total)
From `/tahoe/drive_3/ANALYSIS/analysis_190/Data/20250618.tahoe_embeddings_70M_DC_split_assignments.parquet` - test split perturbations including drugs like Paclitaxel, Dexamethasone, Gefitinib, etc. at doses 005, 05, 50.

---
# After training on full data

## Command Running in Background
```bash
tmux new-session -d -s state_inference 'state tx infer --adata dmso_controls_expanded_for_inference.h5ad --model_dir train_state_tx/experiments/tahoe_state_tx_20250709_214850 --embed_key embedding --pert_col drug_dose --output dmso_controls_predicted_all_perturbations.20250710.h5ad 2>&1 | tee state_inference.log'
```


---
Trained models: 
- Code/train_state_tx/experiments/tahoe_state_tx_20250707_231229 -- Trained using 1M sample MosaicFM embeddings
- Code/train_state_tx/experiments/tahoe_state_tx_20250709_214850 -- Trained using 95M (all) MosaicFM embeddings
- Code/train_state_tx/experiments/tahoe_state_tx_20250710_205911_state_emb_1M_sample -- Trained using 1M sample SE embeddings


---

```bash
tmux new-session -d -s state_inference 'state tx infer --adata ../Data/20250321.Tahoe.full_filtered.1M_sample_with_state_embs.minimal_split_outputs/dmso_controls_expanded_for_inference.h5ad --model_dir train_state_tx/experiments/tahoe_state_tx_20250710_205911_state_emb_1M_sample --embed_key X_state --pert_col drug_dose --output ../Data/20250321.Tahoe.full_filtered.1M_sample_with_state_embs.minimal_split_outputs/dmso_controls_predicted_all_perturbations.20250711.h5ad 2>&1 | tee state_inference.log'
```

---

```bash
bash run_state_tx_inference.sh -m /tahoe/drive_3/ANALYSIS/analysis_190/Code/train_state_tx/experiments/tahoe_state_tx_20250709_214850 -e mosaicfm-70m-merged ../../Data/20250618.tahoe_embeddings_70M_split_outputs/dmso_controls_expanded_for_inference.h5ad ../../Data/20250618.tahoe_embeddings_70M_split_outputs/dmso_controls_predicted_all_perturbations.h5ad
```

```bash
python evaluate_transport_mmd_h5ad_test.py --test-dataset ../../Data/20250618.tahoe_embeddings_70M_split_outputs/test_subset.h5ad --transported-data ../../Data/20250618.tahoe_embeddings_70M_split_outputs/dmso_controls_predicted_all_perturbations.h5ad --dmso-controls ../../Data/20250618.tahoe_embeddings_70M_split_outputs/dmso_controls.h5ad --embedding-key mosaicfm-70m-merged --output-dir ../../Data/20250618.tahoe_embeddings_70M_split_outputs/mmd_evaluation_results
```


--- After training for 400,000 steps instead of 40,000 for HVG

```bash
bash run_state_tx_inference.sh -m /tahoe/drive_3/ANALYSIS/analysis_190/Code/train_state_tx/experiments/tahoe_state_tx_20250719_010325_hvg_full -e X_hvg ../../Data/20250711.tahoe.hvg_split_outputs/dmso_controls_expanded_for_inference.h5ad ../../Data/20250711.tahoe.hvg_split_outputs/dmso_controls_predicted_all_perturbations.h5ad
```

```bash
python evaluate_transport_mmd_h5ad_test.py --test-dataset ../../Data/20250711.tahoe.hvg_split_outputs/test_subset.h5ad --transported-data ../../Data/20250711.tahoe.hvg_split_outputs/dmso_controls_predicted_all_perturbations.h5ad --dmso-controls ../../Data/20250711.tahoe.hvg_split_outputs/dmso_controls.h5ad --embedding-key X_hvg --output-dir ../../Data/20250711.tahoe.hvg_split_outputs/mmd_evaluation_results
```

```bash
python pearson_delta_evaluation.py --predicted ../../Data/20250711.tahoe.hvg_split_outputs/dmso_controls_predicted_all_perturbations.h5ad --test ../../Data/20250711.tahoe.hvg_split_outputs/test_subset.h5ad --dmso-controls ../../Data/20250711.tahoe.hvg_split_outputs/dmso_controls_expanded_for_inference.h5ad --output-dir ../../Data/20250711.tahoe.hvg_split_outputs/pearson_delta_results
```


--- After training on MFM with 100,000 steps

```bash
bash run_state_tx_inference.sh -m /tahoe/drive_3/ANALYSIS/analysis_190/Code/train_state_tx/experiments/tahoe_state_tx_20250722_204424_mfm_full -e mosaicfm-70m-merged ../../Data/20250618.tahoe_embeddings_70M_split_outputs/dmso_controls_expanded_for_inference.h5ad ../../Data/20250618.tahoe_embeddings_70M_split_outputs/dmso_controls_predicted_all_perturbations.h5ad
```

```bash
python evaluate_transport_mmd_h5ad_test.py --test-dataset ../../Data/20250618.tahoe_embeddings_70M_split_outputs/test_subset.h5ad --transported-data ../../Data/20250618.tahoe_embeddings_70M_split_outputs/dmso_controls_predicted_all_perturbations.h5ad --dmso-controls ../../Data/20250618.tahoe_embeddings_70M_split_outputs/dmso_controls.h5ad --embedding-key mosaicfm-70m-merged --output-dir ../../Data/20250618.tahoe_embeddings_70M_split_outputs/mmd_evaluation_results
```


--- Training on HVG with settings based on Abhi's email

```bash
./train_tahoe_state_tx_20250723.sh
```

```bash
bash run_state_tx_inference.sh -m /tahoe/drive_3/ANALYSIS/analysis_190/Code/train_state_tx/experiments/tahoe_state_tx_20250723_183359_hvg_full -e X_hvg ../../Data/20250711.tahoe.hvg_split_outputs/dmso_controls_expanded_for_inference.h5ad ../../Data/20250711.tahoe.hvg_split_outputs/dmso_controls_predicted_all_perturbations.h5ad
```

```bash
python evaluate_transport_mmd_h5ad_test.py --test-dataset ../../Data/20250711.tahoe.hvg_split_outputs/test_subset.h5ad --transported-data ../../Data/20250711.tahoe.hvg_split_outputs/dmso_controls_predicted_all_perturbations.h5ad --dmso-controls ../../Data/20250711.tahoe.hvg_split_outputs/dmso_controls.h5ad --embedding-key X_hvg --output-dir ../../Data/20250711.tahoe.hvg_split_outputs/mmd_evaluation_results
```

```bash
python pearson_delta_evaluation.py --predicted ../../Data/20250711.tahoe.hvg_split_outputs/dmso_controls_predicted_all_perturbations.h5ad --test ../../Data/20250711.tahoe.hvg_split_outputs/test_subset.h5ad --dmso-controls ../../Data/20250711.tahoe.hvg_split_outputs/dmso_controls_expanded_for_inference.h5ad --output-dir ../../Data/20250711.tahoe.hvg_split_outputs/pearson_delta_results
```


--- Using pre-trained ST-Tahoe

```bash
# I think there's a bug here where DMSO_TF is assigned to control instead of DMSO_TF_00.
# Since DMSO_TF is not in the one-hot encoder, index 0 gets used instead.
bash run_state_tx_inference.sh -m /tahoe/drive_3/ANALYSIS/analysis_190/Code/train_state_tx/pretrained/ST-Tahoe -e X_hvg ../../Data/20250711.tahoe.hvg_split_outputs/dmso_controls_expanded_for_inference.h5ad ../../Data/20250711.tahoe.hvg_split_outputs/dmso_controls_predicted_all_perturbations.h5ad
```

```bash
python evaluate_transport_mmd_h5ad_test.py --test-dataset ../../Data/20250711.tahoe.hvg_split_outputs/test_subset.h5ad --transported-data ../../Data/20250711.tahoe.hvg_split_outputs/dmso_controls_predicted_all_perturbations.h5ad --dmso-controls ../../Data/20250711.tahoe.hvg_split_outputs/dmso_controls.h5ad --embedding-key X_hvg --output-dir ../../Data/20250711.tahoe.hvg_split_outputs/mmd_evaluation_results
```

```bash
python pearson_delta_evaluation.py --predicted ../../Data/20250711.tahoe.hvg_split_outputs/dmso_controls_predicted_all_perturbations.h5ad --test ../../Data/20250711.tahoe.hvg_split_outputs/test_subset.h5ad --dmso-controls ../../Data/20250711.tahoe.hvg_split_outputs/dmso_controls_expanded_for_inference.h5ad --output-dir ../../Data/20250711.tahoe.hvg_split_outputs/pearson_delta_results
```


--- Training on HVG, with batch = perturbation

```bash
./train_tahoe_state_tx_20250723.sh
```

```bash
bash run_state_tx_inference.sh -m /tahoe/drive_3/ANALYSIS/analysis_190/Code/train_state_tx/experiments/tahoe_state_tx_20250724_163406_hvg_full -e X_hvg ../../Data/20250711.tahoe.hvg_split_outputs/dmso_controls_expanded_for_inference.h5ad ../../Data/20250711.tahoe.hvg_split_outputs/dmso_controls_predicted_all_perturbations.h5ad
```


--- Train on non-log HVG (as per https://github.com/ArcInstitute/state/issues/132)

```bash
python streaming_dataset2anndata.py ../../Data/20250711.tahoe.hvg.no_log.h5ad --no-log
```

This got interrupted during memory crash. But would have taken a long time. Rethinking.

```bash
python scripts/unlog_h5ad_inplace.py
```

```bash
./train_tahoe_state_tx_20250723.sh
```

```bash
python split_test_data.py /tahoe/drive_3/ANALYSIS/analysis_190/Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged.h5ad
```

```bash
python expand_dmso_for_inference.py /tahoe/drive_3/ANALYSIS/analysis_190/Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_split_outputs/ --embedding-key X_hvg
```


### 20250731
```bash
bash run_state_tx_inference.sh -m /tahoe/drive_3/ANALYSIS/analysis_190/Code/train_state_tx/experiments/tahoe_state_tx_20250726_015550_nonlog_hvg_full -e X_hvg ../../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_split_outputs/dmso_controls_expanded_for_inference.h5ad ../../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_split_outputs/dmso_controls_predicted_all_perturbations.h5ad
```

### Skip
```bash
python evaluate_transport_mmd_h5ad_test.py --test-dataset ../../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_split_outputs/test_subset.h5ad --transported-data ../../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_split_outputs/dmso_controls_predicted_all_perturbations.h5ad --dmso-controls ../../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_split_outputs/dmso_controls.h5ad --embedding-key X_hvg --output-dir ../../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_split_outputs/mmd_evaluation_results
```

### Skip
```bash
python pearson_delta_evaluation.py --predicted ../../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_split_outputs/dmso_controls_predicted_all_perturbations.h5ad --test ../../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_split_outputs/test_subset.h5ad --dmso-controls ../../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_split_outputs/dmso_controls_expanded_for_inference.h5ad --output-dir ../../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_split_outputs/pearson_delta_results
```

### Skip
```bash
python pearson_delta_evaluation.py --log-scale-inputs --predicted ../../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_split_outputs/dmso_controls_predicted_all_perturbations.h5ad --test ../../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_split_outputs/test_subset.h5ad --dmso-controls ../../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_split_outputs/dmso_controls_expanded_for_inference.h5ad --output-dir ../../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_split_outputs/pearson_delta_results_logged
```

### 20250731 -- Complains about mismatches
```bash
cell-eval run --adata-pred ../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_split_outputs/dmso_controls_predicted_all_perturbations.h5ad --adata-real ../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_split_outputs/test_subset.h5ad --control-pert DMSO_TF_00 --pert-col drug_dose --celltype-col cell_line --embed-key X_hvg --outdir ../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_split_outputs/cell-eval-outdir --num-threads 1 --skip-metrics mse,mae,mse_delta,mae_delta,pearson_edistance,clustering_agreement,pr_auc,roc_auc,de_spearman_sig,de_direction_match,de_spearman_lfc_sig,de_sig_genes_recall,de_nsig_counts
```

### 20250731 -- Because mismatches
```bash
python /tahoe/drive_3/ANALYSIS/analysis_190/mmd_state_pipeline/filter_h5ad_perturbations.py --predicted /tahoe/drive_3/ANALYSIS/analysis_190/Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_split_outputs/dmso_controls_predicted_all_perturbations.h5ad --reference /tahoe/drive_3/ANALYSIS/analysis_190/Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_split_outputs/test_subset.h5ad --output /tahoe/drive_3/ANALYSIS/analysis_190/Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_split_outputs/dmso_controls_predicted_filtered.h5ad
```

### 20250731 -- Even though these tests are skipped, it still needs to do DE at init
```bash
cell-eval run --adata-pred ../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_split_outputs/dmso_controls_predicted_filtered.h5ad --adata-real ../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_split_outputs/test_subset.h5ad --control-pert DMSO_TF_00 --pert-col drug_dose --celltype-col cell_line --embed-key X_hvg --outdir ../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_split_outputs/cell-eval-outdir --num-threads 1 --skip-metrics mse,mae,mse_delta,mae_delta,pearson_edistance,clustering_agreement,pr_auc,roc_auc,de_spearman_sig,de_direction_match,de_spearman_lfc_sig,de_sig_genes_recall,de_nsig_counts
```

### 20250731
```bash
python cell-eval/pearson_delta_only.py \
    --adata-pred ../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_split_outputs/dmso_controls_predicted_filtered.h5ad \
    --adata-real ../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_split_outputs/test_subset.h5ad \
    --control-pert DMSO_TF_00 \
    --pert-col drug_dose \
    --celltype-col cell_line \
    --embed-key-real X_hvg \
    --embed-key-pred model_preds \
    --outdir ../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_split_outputs/pearson-delta-only-results
```


--- Use original Arc split and 300,000 steps

```bash
./train_tahoe_state_tx_20250723.sh
```

### 20250731
```bash
python convert_toml_to_parquet_splits.py
```

### 20250731
```bash
python split_test_data.py /tahoe/drive_3/ANALYSIS/analysis_190/Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged.h5ad --split-file /tahoe/drive_3/ANALYSIS/analysis_190/Code/train_state_tx/tahoe_5_holdout_generalization_split_assignments.parquet --output-dir /tahoe/drive_3/ANALYSIS/analysis_190/Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_tahoe5holdout_split_outputs
```

### 20250731
```bash
python expand_dmso_for_inference.py /tahoe/drive_3/ANALYSIS/analysis_190/Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_tahoe5holdout_split_outputs/ --split-file /tahoe/drive_3/ANALYSIS/analysis_190/Code/train_state_tx/tahoe_5_holdout_generalization_split_assignments.parquet --embedding-key X_hvg
```

### 20250731
```bash
bash run_state_tx_inference.sh -m /tahoe/drive_3/ANALYSIS/analysis_190/Code/train_state_tx/experiments/tahoe_state_tx_20250729_201816_nonlog_hvg_full -e X_hvg ../../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_tahoe5holdout_split_outputs/dmso_controls_expanded_for_inference.h5ad ../../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_tahoe5holdout_split_outputs/dmso_controls_predicted_all_perturbations.h5ad
```

### 20250731 (will it complain about mismatches?) Yes
```bash
python cell-eval/pearson_delta_only.py \
    --adata-pred ../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_tahoe5holdout_split_outputs/dmso_controls_predicted_all_perturbations.h5ad \
    --adata-real ../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_tahoe5holdout_split_outputs/test_subset.h5ad \
    --control-pert DMSO_TF_00 \
    --pert-col drug_dose \
    --celltype-col cell_line \
    --embed-key-real X_hvg \
    --embed-key-pred model_preds \
    --outdir ../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_tahoe5holdout_split_outputs/pearson-delta-only-results
```

### 20250731 -- Because mismatches
```bash
python /tahoe/drive_3/ANALYSIS/analysis_190/mmd_state_pipeline/filter_h5ad_perturbations.py --predicted /tahoe/drive_3/ANALYSIS/analysis_190/Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_tahoe5holdout_split_outputs/dmso_controls_predicted_all_perturbations.h5ad --reference /tahoe/drive_3/ANALYSIS/analysis_190/Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_tahoe5holdout_split_outputs/test_subset.h5ad --output /tahoe/drive_3/ANALYSIS/analysis_190/Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_tahoe5holdout_split_outputs/dmso_controls_predicted_filtered.h5ad
```

### 20250731
```bash
python cell-eval/pearson_delta_only.py \
    --adata-pred /tahoe/drive_3/ANALYSIS/analysis_190/Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_tahoe5holdout_split_outputs/dmso_controls_predicted_filtered.h5ad \
    --adata-real ../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_tahoe5holdout_split_outputs/test_subset.h5ad \
    --control-pert DMSO_TF_00 \
    --pert-col drug_dose \
    --celltype-col cell_line \
    --embed-key-real X_hvg \
    --embed-key-pred model_preds \
    --outdir ../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_tahoe5holdout_split_outputs/pearson-delta-only-results
```

### 20250801
```bash
python evaluate_transport_mmd_h5ad_test.py --test-dataset ../../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_tahoe5holdout_split_outputs/test_subset.h5ad --transported-data ../../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_tahoe5holdout_split_outputs/dmso_controls_predicted_all_perturbations.h5ad --dmso-controls ../../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_tahoe5holdout_split_outputs/dmso_controls.h5ad --embedding-key X_hvg --output-dir ../../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_tahoe5holdout_split_outputs/mmd_evaluation_results
```

### 20250801
```bash
python pearson_delta_evaluation.py --predicted ../../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_tahoe5holdout_split_outputs/dmso_controls_predicted_all_perturbations.h5ad --test ../../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_tahoe5holdout_split_outputs/test_subset.h5ad --dmso-controls ../../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_tahoe5holdout_split_outputs/dmso_controls_expanded_for_inference.h5ad --output-dir ../../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_tahoe5holdout_split_outputs/pearson_delta_results
```

### 20250801
```bash
python pearson_delta_evaluation.py --log-scale-inputs --predicted ../../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_tahoe5holdout_split_outputs/dmso_controls_predicted_all_perturbations.h5ad --test ../../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_tahoe5holdout_split_outputs/test_subset.h5ad --dmso-controls ../../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_tahoe5holdout_split_outputs/dmso_controls_expanded_for_inference.h5ad --output-dir ../../Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.unlogged_tahoe5holdout_split_outputs/pearson_delta_results_logged
```
