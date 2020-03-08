for y_interest in Residual Prognostic
do
    for res in 0 1 2
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
                                           --PROJECT_NAME nature --PROJECT_VERSION 2-0 \
                                            --resolution $res --y_interest ${y_interest} \
                                            --label /mnt/data3/pnaylor/CellularHeatmaps/outputs/label_20_02_20.csv \
                                            --input_tiles "./outputs/nature_2-0/tiling/${res}/mat_pca/" \
                                            --mean ./outputs/nature_2-0/tiling/${res}/pca_mean/mean.npy
    done
done

