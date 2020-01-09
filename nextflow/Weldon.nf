#!/usr/bin/env nextflow

params.PROJECT_NAME = "TEST"
params.PROJECT_VERSION = "1-0"
params.resolution = "2"
r = params.resolution
params.class_type = "residuum"

// Folders
output_folder = "./outputs/${params.PROJECT_NAME}_${params.PROJECT_VERSION}"
output_folder = "${output_folder}/Weldon_R${r}"

// label
params.label = "/Users/naylorpeter/Documents/Histopathologie/labels.csv"
label = file(params.label)

results_table = "/path/to/the/csv"
inner_fold =  10
weldon_training = file("python_files/Weldon/main.py")
inner_cross_validation_number = 2
gaussian_noise = [0]//, 1]
batch_size = 16
epochs = 40
repeat = 4
params.size = 5000
size = params.size
number_of_folds = 10 
seed = 42

/* Channels definitions */
params.input = "./outputs/${params.PROJECT_NAME}_${params.PROJECT_VERSION}/tiling/${r}/mat/"
encoded_bags = file(params.input + "/*.npy")
mean_file = Channel.from(file("outputs/prepare_patients_res_${params.resolution_level}/${weight}/mean_file_flatten/mean.npy"))
patients = Channel.from(encoded_bags) 

process TrainingWeldon {
:weight
    publishDir "${output_model_folder}", pattern: "*.h5", overwrite: True
    publishDir "${output_results_folder}", pattern: "*.csv", overwrite: True
    memory { 30.GB + 5.GB * (task.attempt - 1) }
    errorStrategy 'retry'
    maxRetries 6
    queue 'gpu-cbio'

    input:
    file images from patients .collect() 
    file mean from mean_file
    each fold from in 1..number_of_folds 

    output:
    tuple val("${fold}"), file("*.csv") into results_weldon

    script:
    python_script = file("./python/Weldon/main.py")
    output_model_folder = "${output_folder}/models/test_fold_${fold}/"
    output_results_folder = "${output_folder}/results/test_fold_${fold}/"

    /* Mettre --table --repeat --class_type en valeur par défaut ? */
    """
    module load cuda10.0
    python $python_script --mean_name $mean_file  \ 
                          --path $patients \ 
                          --seed $seed \ 
                          --table $results_table \ 
                          --batch_size $batch_size \ 
                          --epochs $epochs \ 
                          --size $size \ 
                          --fold_test $fold \ 
                          --inner_cross_validation_number $inner_cross_validation_number \ 
                          --class_type $param.class_type \ 
                          --repeat $repeat \ 
                          --n_folds $number_of_folds \
                          --y_variable $params.y_variable
    """
}

