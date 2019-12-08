nextflow run nextflow/Tiling-encoding.nf -resume -c ~/.nextflow/config -profile mines \
                                         --tiff_location ../Data/Combined_Biopsy \
                                         --tissue_bound_annot ../Data/Combined_Biopsy/combined_tissue_segmentation