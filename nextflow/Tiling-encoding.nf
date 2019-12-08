#!/usr/bin/env nextflow

params.PROJECT_NAME = "TEST"
params.PROJECT_VERSION = "1-0"
output_folder = "./outputs/${params.PROJECT_NAME}_${params.PROJECT_VERSION}"

params.tiff_location = "../Data/Biopsy" // tiff files to process
params.tissue_bound_annot = "../Data/Biopsy/tissue_segmentation" // xml folder containing tissue segmentation mask for each patient

// input file
tiff_files = file(params.tiff_location + "/*.tiff")
boundaries_files = file(params.tissue_bound_annot)

// input parameter
params.weights = "imagenet"
weights = params.weights

levels = [0, 1, 2]

process WsiTilingEncoding {
    publishDir "${output_process_mean}", overwrite: true, pattern: "${name}.npy"
    publishDir "${output_process_mat}", overwrite: true, pattern: "${name}_mean.npy"
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
    file("${name}.npy")
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
              .set { all_patient_means }


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