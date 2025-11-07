We need a script that takes a .h5ad file as input, and the name of a categorical
column in .obs. It should then create a subfolder in the same directory named
'by_{category_name}', and create one separate .h5ad with the data for that category
in it. The file name of the output .h5ad should reflect the cateogry.

Note that category names can contain special characters and other feature that can
incompatible with valid file names. So make sure to include handling of that
converts category names with spaces or slashes etc to ones that are copmatible
with file names when making the file names.

The script need to be efficient. Take inspiration from the other scripts in this
directory about efficiently working with h5ads making use of raw access using h5py.

Please include detailed progress tracking with timestamping as we typically do
in order to keep track of progress and identify bottlenecks.

