#!/usr/bin/env nextflow

params.PROJECT_NAME = "TEST"
params.PROJECT_VERSION = "1-0"
params.resolution = "2"
r = params.resolution
params.class_type = "residuum"
params.y_variable = "Residual"

// Folders
input_folder = "./outputs/${params.PROJECT_NAME}_${params.PROJECT_VERSION}"
output_folder = "${input_folder}/Model_1S_R${r}"

// label
params.label = "/mnt/data3/pnaylor/CellularHeatmaps/outputs/label.csv"
label = file(params.label)
params.y_interest = "Residual"
y_interest = params.y_interest

params.input_tiles = "${input_folder}/tiling/${r}/mat/*.npy"
input_tiles = file(params.input_tiles)
params.mean = "${input_folder}/tiling/${r}/mean/mean.npy"
mean_tiles = file(params.mean)



inner_fold =  5
inner_cross_validation_number = 2
gaussian_noise = [0]//, 1]
batch_size = 16
epochs = 40
repeat = 4
params.size = 5000
size = params.size
number_of_folds = 10 
seed = 42    
pooling_layer = ["avg"]//, "max"]



//mean_file = mean_file .view()

process Training_Model_1S {
    publishDir "${output_model_folder}", pattern: "*.h5", overwrite: true
    publishDir "${output_results_folder}", pattern: "*.csv", overwrite: true
    memory { 30.GB + 5.GB * (task.attempt - 1) }
    errorStrategy 'retry'
    maxRetries 6
    queue 'gpu-cbio'
    clusterOptions "--gres=gpu:1"

    input:
    file images from encoded_bags .collect() 
    file mean from mean_file
    each fold from 1..number_of_folds 

    output:
    tuple val("${fold}"), file("*.csv") into results_weldon
    file("*.h5")

    script:
    python_script = file("./python/nn/1S_main.py")
    output_model_folder = file("${output_folder}/models/")
    output_results_folder = file("${output_folder}/results/")

    /* Mettre --table --repeat --class_type en valeur par défaut ? */
    """
    module load cuda10.0
    python $python_script --mean_name ${params.input_mean} \
                          --path "${params.input_tiles}" \
                          --seed $seed \
                          --table $labels \
                          --batch_size $batch_size \
                          --epochs $epochs \
                          --size $size \
                          --fold_test $fold \
                          --inner_cross_validation_number $inner_cross_validation_number \
                          --class_type $params.class_type \
                          --repeat $repeat \
                          --n_fold $number_of_folds \
                          --y_variable $params.y_variable
    """
}

