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
			nextflow run nextflow/Hard_labelling.nf -resume -c ~/.nextflow/config -profile $3 \
											--PROJECT_NAME $4 --PROJECT_VERSION $5 \
											--resolution $res --y_interest ${y_interest} \
											--label $6 \
											--input_tiles ./outputs/tiling/${res}/mat_pca/ \
											--mean ./outputs/tiling/${res}/pca_mean/mean.npy
		done
	done