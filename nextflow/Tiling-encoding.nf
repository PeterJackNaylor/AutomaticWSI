#!/usr/bin/env nextflow

params.PROJECT_NAME = "TEST"
params.PROJECT_VERSION = "1-0"
output_folder = "./outputs/${params.PROJECT_NAME}_${params.PROJECT_VERSION}"

params.tiff_location = "/mnt/data3/pnaylor/Data/Biopsy" // tiff files to process
params.tissue_bound_annot = "/mnt/data3/pnaylor/Data/Biopsy/tissue_segmentation" // xml folder containing tissue segmentation mask for each patient

// input file
tiff_files = file(params.tiff_location + "/*.tiff")
boundaries_files = file(params.tissue_bound_annot)

params.label = "/mnt/data3/pnaylor/CellularHeatmaps/outputs/label.csv"
label = file(params.label)

// input parameter
params.weights = "imagenet"
weights = params.weights

levels = [0, 1, 2]

process WsiTilingEncoding {
    publishDir "${output_process_mat}", overwrite: true, pattern: "${name}.npy"
    publishDir "${output_process_mean}", overwrite: true, pattern: "${name}_mean.npy"
    publishDir "${output_process_info}", overwrite: true, pattern: "${name}_info.txt"
    publishDir "${output_process_visu}", overwrite: true, pattern: "${name}_visu.png"

    queue "gpu-cbio"
    clusterOptions "--gres=gpu:1"
    maxForks 16
    memory '20GB'
    
    input:
    file slide from tiff_files
    each level from levels
    
    output:
    set val("$level"), file("${name}.npy") into bags
    set val("$level"), file("${name}_mean.npy") into mean_patient
    file("${name}_info.txt")
    file("${name}_visu.png")

    script:
    py = file("./python/preparing/process_one_patient.py")
    name = slide.baseName
    xml_file = file(boundaries_files + "/${name}.xml")
    output_process_mean = "${output_folder}/tiling/${level}/mean"
    output_process_mat = "${output_folder}/tiling/${level}/mat"
    output_process_info = "${output_folder}/tiling/${level}/info"
    output_process_visu = "${output_folder}/tiling/${level}/visu"
    """
    module load cuda10.0
    python $py --slide $slide \
               --xml_file $xml_file \
               --analyse_level $level \
               --weight $weights
    """
}

mean_patient  .groupTuple() 
              .into { all_patient_means ; all_patient_means2 }


process ComputeGlobalMean {
    publishDir "${output_process}", overwrite: true
    memory { 10.GB }
    input:
    set level, file(_) from all_patient_means
    output:
    file('mean.npy')

    script:
    compute_mean = file('./python/preparing/compute_mean.py')
    output_process = "${output_folder}/tiling/$level/mean/"

    """
    python $compute_mean 
    """
}

y = ["Residual", "Prognostic"]
process RandomForestlMean {
   publishDir "${output_process}", overwrite: true
   memory { 10.GB }
   cpus 8
   input:
   set level, file(_) from all_patient_means2
   file lab from label
   each y_interest from y

   output:
   file('*.txt')

   script:
   compute_rf = file("./python/naive_rf/compute_rf.py")
   output_process = "${output_folder}/naive_rf_${level}/${y_interest}"

   """
   python $compute_rf --label $label \
                      --inner_fold 5 \
                      --y_interest $y_interest \
                      --cpu 8
   """
}

// keep bags_1 to collect them and process the PCA
// bags_2 is a copy, to after compute the transformed tiles
// bags_per_level = [($level, (*.npy))], for each level the whole tiles files.
bags .into{ bags_1; bags_2 }
bags_1 .groupTuple()
     .set{ bags_per_level }

process Incremental_PCA {
    publishDir "${output_process_pca}", overwrite: true
    memory '60GB'
    cpus '16'

    input:
    tuple level, files from bags_per_level
    
    output:
    tuple level, file("*.joblib") into results_PCA

    script:
    output_process_pca = "${output_folder}/tiling/${level}/pca/"
    input_tiles = file("${output_folder}/tiling/${level}/mat")
    python_script = file("./python/preparing/pca_partial.py")

    """
    python $python_script --path ${input_tiles}
    """
}

// files_to_transform = [ ($level, pca_res_level, f1.npy ), ($level, pca_res_level, f2.npy), ... ]
// begins to get filled as soon as a level has been treated by the PCA.
results_PCA .combine(bags_2, by: 0) 
            .set { files_to_transform } 

process Transform_Tiles {

    publishDir "${output_mat_pca}", overwrite: true
    memory '60GB'

    input:
    tuple level, file(pca), tile from files_to_transform

    output:
    tuple level, file("*.npy") into transform_tiles

    script:
    output_mat_pca = "${output_folder}/tiling/$level/mat_pca"
    python_script = file("./python/preparing/transform_tile.py")

    """
    python ${python_script} --path $tile --pca $pca
    """
}

transform_tiles  .groupTuple() 
              .set { transform_tiles_per_level }

process ComputePCAGlobalMean {
    publishDir "${output_pca_mean}", overwrite: true
    memory { 10.GB }

    input:
    set level, file(_) from transform_tiles_per_level
    output:
    file('mean.npy')

    script:
    output_pca_mean = "${output_folder}/tiling/$level/pca_mean"
    compute_mean_pca = file('./python/preparing/compute_mean_pca.py')

    """
    echo $level
    python $compute_mean_pca
    """
}
