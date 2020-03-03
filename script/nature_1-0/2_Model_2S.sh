

for y_interest in Residual Prognostic
do
    for res in 0 1 2
    do
        echo "####################################################################"
        echo 
        echo "########### Doing ${y_interest} at ${res} ###############"
        echo 
        echo "####################################################################"
        nextflow run nextflow/Model_2S.nf -resume -c ~/.nextflow/config -profile mines \
                                           --PROJECT_NAME nature --PROJECT_VERSION 2-0 \
                                            --resolution $res --y_interest ${y_interest} \
                                            --label /mnt/data3/pnaylor/CellularHeatmaps/outputs/label_20_02_20.csv
    done
done
