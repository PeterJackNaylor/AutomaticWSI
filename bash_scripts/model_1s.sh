for y_interest in $1
do
    for res in $2
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
        nextflow run nextflow/Model_nn.nf -resume -c ~/.nextflow/config -profile $3 \
                                        --PROJECT_NAME CIGA --PROJECT_VERSION $5 \
                                        --resolution $res --y_interest ${y_interest} \
                                        --label $6 --size $size \
                                        --input_tiles ./outputs/ciga/tiling/${res}/mat_pca/ \
										--mean ./outputs/ciga/tiling/${res}/mean.npy
                                        # --mean ./outputs/$4_$5/tiling/${res}/pca_mean/mean.npy
                                        #./outputs/$4_$5/tiling/${res}/mat_pca/ \
                                        
    done
done

for y_interest in $1
do
    for res in $2
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
        nextflow run nextflow/Model_nn.nf -resume -c ~/.nextflow/config -profile $3 \
                                        --PROJECT_NAME MOCO --PROJECT_VERSION $5 \
                                        --resolution $res --y_interest ${y_interest} \
                                        --label $6 --size $size \
                                        --input_tiles ./outputs/moco/tiling/${res}/mat_pca/ \
										--mean ./outputs/moco/tiling/${res}/mean.npy
                                        # --mean ./outputs/$4_$5/tiling/${res}/pca_mean/mean.npy
                                        #./outputs/$4_$5/tiling/${res}/mat_pca/ \
                                        
    done
done