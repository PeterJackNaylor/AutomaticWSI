
nextflow run nextflow/Tiling-encoding.nf -resume -c ~/.nextflow/config -profile mines \
                                         --tiff_location ../Data/nature_medecine_biop/ \
                                         --PROJECT_NAME nature --PROJECT_VERSION 2-0 \
                                         --tissue_bound_annot ../Data/nature_medecine_biop/tissue_segmentation --label /mnt/data3/pnaylor/CellularHeatmaps/outputs/label_20_02_20.csv
