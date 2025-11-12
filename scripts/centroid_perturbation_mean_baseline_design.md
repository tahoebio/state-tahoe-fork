# Pearson Delta correlation baseline using centroids

We want to get baseline values of what the Pearson Delta correlation is in a held out
test set when using the mean of perturbations from the training set. We need to create
a script for this.

Which perturbations and cell lines combinations that are part of the test and
validation sets are defined in TOML files, such as the file
@/tahoe/drive_3/ANALYSIS/analysis_190/Code/state-tahoe-fork/scripts/train_state_tx/tahoe_5_holdout/generalization_converted_cell_lines.toml

The other input is a single-plate h5ad with centroid information. Centroids were
calculated by taking the means of all the cells in a given grouping. An example of
centroid h5ad can be found at
@/tahoe/drive_3/ANALYSIS/analysis_190/Data/state_input_merged/by_plate_centroids/plate_plate1.h5ad
These files are small and easy to work with.

The field of interest in the h5ad is 'X_hvg' in .obsm.

Let mu_c^ctrl be the centroid for control in cell type c.
Let mu_c,p^pert be the centroid for perturbation p in cell type c.
Create the cell-type offsets delta_c,p = mu_c,p^pert - mu_c^ctrl
Average over all cell types in the training set that contain p to create delta_p

Given a test cell type t and a perturbation label p, make the prediction
hat(x) = mu_t^ctrl + delta_p. This is done for held out combination of cell line
and perturbation. This is compared to the true delta_t,p calculated from data using
Pearson correlation.

As arguments, the script needs to take the name of the cell type (context)
column in .obs, the name of the perturbation column in .obs, and the name of the
perturbation control condition in the perturbation column of .obs.

Special handling is needed when the control condition is 'DMSO_TF'. You can see example
of this in the script
@/tahoe/drive_3/ANALYSIS/analysis_190/Code/cell-eval/pearson_delta_only.py. The script
generally shows how to take the arguments from the user. (It works on single cells 
though, so we can't use the Pearson delta correlation computation framework in it)

