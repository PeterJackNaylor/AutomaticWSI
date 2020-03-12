#!/usr/bin/env nextflow

params.PROJECT_NAME = "TEST"
params.PROJECT_VERSION = "1-0"
output_folder = "./outputs/${params.PROJECT_NAME}_${params.PROJECT_VERSION}"
params.resolution = "2"
r = params.resolution
output_folder = "${output_folder}/model_2S_R${r}"

// label
params.label = "/mnt/data3/pnaylor/CellularHeatmaps/outputs/label.csv"
label = file(params.label)
params.y_interest = "Residual"
y_interest = params.y_interest

// raw input
params.input = "./outputs/${params.PROJECT_NAME}_${params.PROJECT_VERSION}/tiling/${r}/mat/"
encoded_bags = file(params.input + "/*.npy")

params.inner_fold = 5
inner_fold =  params.inner_fold



nber_ds = 1000
// subsample_patient
subsampling_method = ["uniform", "kmeans"]
// create_cluster_feature_per_patient
n_clusters = [2, 4, 8, 16, 32, 64]
unsupervised_method = ["KMeans"] //, "UMAP"]
// plot cluster
ploting_method = ["UMAP"] // "PCA", "t-SNE", "MDS",
// classification 
classifcation_method = ["RandomForest"]

process SubsamplingTissue {

    publishDir "${output_process}", overwrite: true
    memory { 10.GB * task.attempt }
    errorStrategy 'retry' 

    input:
    file npy from encoded_bags
    each method from subsampling_method

    output:
    set val("${method}"), file("${name}_ds.npy"), file("${npy}") into ds_encoded_bags

    script:
    py_sub = file("./python/model_2S/subsample.py")
    name = npy.baseName
    output_process = "${output_folder}/subsampling/${method}_ds"

    """
    python $py_sub --npy $npy \
                   --method $method \
                   --nber_ds $nber_ds
    """ 
}

ds_encoded_bags  .groupTuple() 
                .set { ds_encoded_bag_grouped }

process TileClassification {
    publishDir "${output_process_mod}", pattern: "models", overwrite: true
    publishDir "${output_process_zi}", pattern: "*.npy", overwrite: true
    publishDir "${output_process_zi}", pattern: "order_zi.pickle", overwrite: true

    memory '100GB'
    // cpus 8

    input:
    set method, file(bag_ds), file(bag) from ds_encoded_bag_grouped
    each k from n_clusters
    each clus from unsupervised_method

    output:
    set val("${method}_${k}_${clus}"), file("models") 
    set val("${method}_${k}_${clus}"), file("tissue_zi.npy"), file("order_zi.pickle") into z_is
    
    script:
    py_tile_classification = file("./python/model_2S/tile_classification.py")
    output_process_mod = "${output_folder}/tile_classification/${method}_${k}_${clus}/models"
    output_process_zi = "${output_folder}/tile_classification/${method}_${k}_${clus}/patient_tissue_zi"

    """
    python $py_tile_classification --path . \
                                   --n_c $k \
                                   --clustering_method $clus \
                                   --cpu 8 \
                                   --seed 42
    """
}

process TissueClassification {

    publishDir "${output_process_visu}", pattern: "*.png", overwrite: true
    publishDir "${output_process_results}", pattern: "*.txt", overwrite: true
    publishDir "${output_process_results}", pattern: "*.csv", overwrite: true
    publishDir "${output_process_results}", pattern: "*.pickle", overwrite: true

    
    // cpus 8

    input:
    set name, file(npy), file(order) from z_is
    each method from classifcation_method
    
    output:
    file("*.txt")
    file("*.pickle")
    file("*.csv")
    file("*.png")

    script:
    py_tissue_classification = file("./python/model_2S/tissue_classification.py")
    output_process_visu = "${output_folder}/tissue_classification//${y_interest}/${name}_${method}/visu"
    output_process_results = "${output_folder}/tissue_classification//${y_interest}/results/"

    // set val("${name}__${method}__${pred}"), file("*__error_scores.txt") into scores
    """
    python $py_tissue_classification --main_name ${name}_${method} \
                                 --method $method \
                                 --label $label \
                                 --y_interest $y_interest \
                                 --dataset $npy \
                                 --order $order \
                                 --cpu 8 \
                                 --inner_fold $inner_fold 
    """
}

// //TO DO: plot extremas in these plots

// process cluster_plots {
//     tag { "Plotting $method on $name, $clus"}
//     publishDir "./outputs/${class_type}/${weight}/model_1/clusterplots_${name}-${clus}-${method}"
//     memory "60GB"
//     cpus 10
//     input:
//     set name, clus, file(patch_file), file(patch_detail), file(patient_pred) from patches_to_cluster
//     each method from ploting_method
//     output:
//     file "*.png"
//     """
//     python $ploting_cluster --method $method --table $patch_file --detail $patch_detail --prediction $patient_pred --name $name
//     """
// }

// process classification {
//     tag { "Performing ${pred} method: ${name}-${clus}-${method} " }
//     publishDir "./outputs/${class_type}_${y_variable}/${weight}/model_1/model_${name}-${clus}-${method}/meta", pattern: "*.csv"
//     publishDir "./outputs/${class_type}_${y_variable}/${weight}/model_1/model_${name}-${clus}-${method}/meta", pattern: "*.pickle"
//     publishDir "./outputs/${class_type}_${y_variable}/${weight}/model_1/model_${name}-${clus}-${method}/scores", pattern: "*.txt"
//     cpus 8
//     input:
//     set name, clus, file(table) from patient_tables
//     each method from classifcation_method
//     each pred from prediction_type
//     output:
//     set file("*_scores.txt"), file("*__fold_scores.pickle"), file("*__pred_y.csv") into all_results
//     // set val("${name}__${method}__${pred}"), file("*__error_scores.txt") into scores
//     """
//     python $classification_model --main_name model_performance  --method $method \\
//                                     --table $table --detail $label_fold --prediction_type $pred \\
//                                     --class_type $class_type --cpus 8 --seed $seed
//     """
// }
