```bash
python /tahoe/drive_3/ANALYSIS/analysis_190/Code/state-tahoe-fork/scripts/centroid_global_mean_baseline.py \
--toml-file /tahoe/drive_3/ANALYSIS/analysis_190/Code/state-tahoe-fork/scripts/train_state_tx/tahoe_5_holdout/zeroshot_generalization_converted_cell_lines.toml \
--centroids-dir /tahoe/drive_3/ANALYSIS/analysis_190/Data/state_input_merged/by_plate_centroids/ \
--output-dir /tahoe/drive_3/ANALYSIS/analysis_190/Code/train_state_tx/experiments/zeroshot/tahoe_globalmean_hvg_full_plate_matched/
```

```bash
python /tahoe/drive_3/ANALYSIS/analysis_190/Code/state-tahoe-fork/scripts/centroid_global_mean_baseline.py \
--toml-file /tahoe/drive_3/ANALYSIS/analysis_190/Code/state-tahoe-fork/scripts/train_state_tx/tahoe_5_holdout/zeroshot_generalization_converted_cell_lines.toml \
--centroids-dir /tahoe/drive_3/ANALYSIS/analysis_190/Data/state_input_merged/by_plate_centroids/ \
--output-dir /tahoe/drive_3/ANALYSIS/analysis_190/Code/train_state_tx/experiments/zeroshot/tahoe_globalmean_hvg_full/ \
--ignore-plate-boundaries
```

```bash
python /tahoe/drive_3/ANALYSIS/analysis_190/Code/state-tahoe-fork/scripts/centroid_perturbation_mean_baseline.py \
--toml-file /tahoe/drive_3/ANALYSIS/analysis_190/Code/state-tahoe-fork/scripts/train_state_tx/tahoe_5_holdout/zeroshot_generalization_converted_cell_lines.toml \
--centroids-dir /tahoe/drive_3/ANALYSIS/analysis_190/Data/state_input_merged/by_plate_centroids/ \
--output-dir /tahoe/drive_3/ANALYSIS/analysis_190/Code/train_state_tx/experiments/zeroshot/tahoe_perturbmean_hvg_full_plate_matched/
```

```bash
python /tahoe/drive_3/ANALYSIS/analysis_190/Code/state-tahoe-fork/scripts/centroid_perturbation_mean_baseline.py \
--toml-file /tahoe/drive_3/ANALYSIS/analysis_190/Code/state-tahoe-fork/scripts/train_state_tx/tahoe_5_holdout/zeroshot_generalization_converted_cell_lines.toml \
--centroids-dir /tahoe/drive_3/ANALYSIS/analysis_190/Data/state_input_merged/by_plate_centroids/ \
--output-dir /tahoe/drive_3/ANALYSIS/analysis_190/Code/train_state_tx/experiments/zeroshot/tahoe_perturbmean_hvg_full/ \
--ignore-plate-boundaries
```

```bash
python /tahoe/drive_3/ANALYSIS/analysis_190/Code/state-tahoe-fork/scripts/centroid_context_mean_baseline.py \
--toml-file /tahoe/drive_3/ANALYSIS/analysis_190/Code/state-tahoe-fork/scripts/train_state_tx/tahoe_5_holdout/zeroshot_generalization_converted_cell_lines.toml \
--centroids-dir /tahoe/drive_3/ANALYSIS/analysis_190/Data/state_input_merged/by_plate_centroids/ \
--output-dir /tahoe/drive_3/ANALYSIS/analysis_190/Code/train_state_tx/experiments/zeroshot/tahoe_contextmean_hvg_full_plate_matched/
```






```bash
state tx predict \
--output_dir . \
--checkpoint 'final.ckpt' \
--profile anndata \
--predict_only
```
```bash
python /tahoe/drive_3/ANALYSIS/analysis_190/Code/cell-eval/pearson_delta_only.py \
--adata-pred adata_pred.h5ad \
--adata-real adata_real.h5ad \
--control-pert DMSO_TF \
--pert-col 'drugname_drugconc' \
--celltype-col cell_line_id \
--allow-discrete \
--outdir ./pearson-delta-only-results
```
```bash
python /tahoe/drive_3/ANALYSIS/analysis_190/Code/cell-eval/pearson_delta_only.py \
--adata-pred adata_pred.h5ad \
--adata-real adata_real.h5ad \
--control-pert DMSO_TF \
--pert-col 'drugname_drugconc' \
--celltype-col cell_line_id \
--allow-discrete \
--group-by plate \
--outdir ./pearson-delta-only-by-plate-results
```





```bash
python /tahoe/drive_3/ANALYSIS/analysis_190/Code/cell-eval/pearson_delta_only.py \
--adata-pred adata_pred.h5ad \
--adata-real adata_real.h5ad \
--control-pert PBS \
--pert-col cytokine \
--celltype-col donor \
--allow-discrete \
--outdir pearson-delta-only-results \
--group-by cell_type
```

```bash
python /tahoe/drive_3/ANALYSIS/analysis_190/Code/cell-eval/pearson_delta_only.py \
--adata-pred adata_pred.h5ad \
--adata-real adata_real.h5ad \
--control-pert PBS \
--pert-col cytokine \
--celltype-col donor \
--allow-discrete \
--outdir ungrouped-pearson-delta-only-results
```




