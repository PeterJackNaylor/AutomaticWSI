
PROJECT_NAME=nature
PROJECT_VERSION=3-0
TIFF_LOCATION=../Data/Biopsy_Nature_3-0/
TISSUE_BOUNDARIES=../Data/Biopsy_Nature_3-0/tissue_segmentation
INNER_FOLD=5
NUMBER_OF_FOLDS=10
LABEL=/mnt/data3/pnaylor/AutomaticWSI/outputs/${PROJECT_NAME}_${PROJECT_VERSION}/label.csv

## tiling
# nextflow run nextflow/Tiling-encoding.nf -resume -c ~/.nextflow/config -profile mines \
#                                          --tiff_location $TIFF_LOCATION \
#                                          --PROJECT_NAME $PROJECT_NAME --PROJECT_VERSION $PROJECT_VERSION \
#                                          --tissue_bound_annot $TISSUE_BOUNDARIES \
#                                          --label $LABEL --inner_fold $INNER_FOLD

## Model 2S
# for y_interest in Residual Prognostic
# do
#     for res in 0 1 2
#     do
#         echo "####################################################################"
#         echo 
#         echo "########### Doing ${y_interest} at ${res} ###############"
#         echo 
#         echo "####################################################################"
#         nextflow run nextflow/Model_2S.nf -resume -c ~/.nextflow/config -profile mines \
#                                            --PROJECT_NAME $PROJECT_NAME --PROJECT_VERSION $PROJECT_VERSION \
#                                             --resolution $res --y_interest ${y_interest} \
#                                             --label $LABEL --inner_fold $INNER_FOLD \
#                                             --input ./outputs/${PROJECT_NAME}_${PROJECT_VERSION}/tiling/${res}/mat_pca
#     done
# done

## Model 1S, owkin, conan and conan++

for y_interest in Residual Prognostic
do
    for res in 1 2 #0
    do
        if [ $res -eq 0 ]
            then
                size=5000
            else
                if [ $res -eq 1 ]
                    then
                        size=3000
                    else
                        size=1000
                fi
        fi
        echo "####################################################################"
        echo 
        echo "########### Doing ${y_interest} at ${res} input size $size ###############"
        echo 
        echo "####################################################################"
        nextflow run nextflow/Model_nn.nf -resume -c ~/.nextflow/config -profile mines \
                                           --PROJECT_NAME $PROJECT_NAME --PROJECT_VERSION $PROJECT_VERSION \
                                            --resolution $res --y_interest ${y_interest} \
                                            --label $LABEL --size ${size} \
                                            --input_tiles "./outputs/${PROJECT_NAME}_${PROJECT_VERSION}/tiling/${res}/mat_pca/" \
                                            --mean ./outputs/${PROJECT_NAME}_${PROJECT_VERSION}/tiling/${res}/pca_mean/mean.npy \
                                            --inner_fold $INNER_FOLD --number_of_folds $NUMBER_OF_FOLDS
    done
done

