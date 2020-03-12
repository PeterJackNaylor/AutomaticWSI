#!/usr/bin/env nextflow

params.PROJECT_NAME = "TEST"
params.PROJECT_VERSION = "1-0"
params.resolution = "2"
r = params.resolution
params.y_interest = "Residual"

// Folders
input_folder = "./outputs/${params.PROJECT_NAME}_${params.PROJECT_VERSION}"
output_folder = "${input_folder}/Model_NN_R${r}"

// label
params.label = "/mnt/data3/pnaylor/CellularHeatmaps/outputs/label.csv"
label = file(params.label)

params.input_tiles = "${input_folder}/tiling/${r}/mat/"
input_tiles = file(params.input_tiles)
params.mean = "${input_folder}/tiling/${r}/mean/mean.npy"
mean_tile = file(params.mean)



inner_fold =  params.inner_fold
gaussian_noise = [0]//, 1]
batch_size = 16
epochs = 40
repeat = 4
params.size = 5000
size = params.size
input_depth = 1024
params.number_of_folds
number_of_folds = params.number_of_folds 
pooling_layer = ["avg"]//, "max"]

model_types = ["model_1S_a", "model_1S_b", "model_1S_c", "model_1S_d", "owkin"]

//mean_file = mean_file .view()

process Training_nn {
    publishDir "${output_model_folder}", pattern: "*.h5", overwrite: true
    publishDir "${output_results_folder}", pattern: "*.csv", overwrite: true
    memory { 30.GB + 5.GB * (task.attempt - 1) }
    // errorStrategy 'retry'
    // maxRetries 6
    cpus 5
    queue 'gpu-cbio'
    clusterOptions "--gres=gpu:1"
    // scratch true
    stageInMode 'copy'

    input:
    file path from input_tiles 
    file mean from mean_tile
    file lab from label
    each fold from 1..number_of_folds
    each model from model_types

    output:
    tuple val("${fold}"), file("*.csv") into results_weldon
    file("*.h5")

    script:
    python_script = file("./python/nn/main.py")
    output_model_folder = file("${output_folder}/${model}/${params.y_interest}/models/")
    output_results_folder = file("${output_folder}/${model}/${params.y_interest}/results/")

    /* Mettre --table --repeat --class_type en valeur par défaut ? */
    """
    module load cuda10.0
    python $python_script --mean_name $mean \
                          --path "${path}/*.npy" \
                          --table $lab \
                          --batch_size $batch_size \
                          --epochs $epochs \
                          --size $size \
                          --fold_test $fold \
                          --repeat $repeat \
                          --y_interest $params.y_interest \
                          --inner_folds $inner_fold \
                          --model $model \
                          --workers 5
    """
}

