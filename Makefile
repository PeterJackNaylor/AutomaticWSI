
PROJECT_NAME = TNBC_PRED  # Project name
VERSION = 1 # Project version
LABEL_FILE = ./outputs/label.csv # Label name
PROFILE = mines #nextflow profile, us it was the mine cluster profile
DATA_PATH = ../Data/TNBC_biopsy/ # /path/to/tiff
TISSUE_SEGMENTATION = ../Data/TNBC_biopsy/tissue_segmentation # /path/to/wsi/tissue/masks
RESOLUTIONS = 0 1 2 # Resolution to investigate
Y_TARGETS = Residual Prognostic # Name of y targets that can be found in LABEL_FILE


$(LABEL_FILE):
	CSV_DATA1 = ../Data/Biopsy_csv/rcb_substrat.csv
	PATH_DATA1 = ../Data/Biopsy_guillaume
	OUT_DATA1 = ../Data/Biopsy_csv/rcb_substrat_after.csv

	XLSX_DATA2 = ../Data/Biopsy_csv/MarickDataSet.xlsx
	PATH_DATA2 = ../Data/Biopsy/
	OUT_DATA2 = ../Data/Biopsy_csv/ftnbc.csv

	python python/label_standardise/substrat_label.py --input_table $(CSV_DATA1) \
													--folder_to_check $(PATH_DATA1) \
													--output_table $(OUT_DATA1)
	python python/label_standardise/ftnbc_label.py --input_table $(XLSX_DATA2) \
												--folder_to_check $(PATH_DATA2) \
												--output_table $(OUT_DATA2)
	python python/label_standardise/create_test_folds.py --substra $(OUT_DATA1) \
														--ftnbc $(OUT_DATA2) \
														--output_name $(LABEL_FILE)
	ln -s  LINK
	ln -sr $(PATH_DATA1)/* $(DATA_PATH)/

encoding: $(LABEL_FILE)
	nextflow run nextflow/Tiling-encoding.nf -resume -c ~/.nextflow/config -profile $(PROFILE) \
											--tiff_location $(DATA_PATH)/ \
											--PROJECT_NAME $(PROJECT_NAME) --PROJECT_VERSION $(VERSION) \
											--tissue_bound_annot $(TISSUE_SEGMENTATION) \
											--label $(LABEL_FILE)

model_2S: $(LABEL_FILE)
	for y_interest in $(Y_TARGETS)
	do
		for res in $(RESOLUTIONS)
		do
			echo "####################################################################"
			echo 
			echo "########### Doing ${y_interest} at ${res} ###############"
			echo 
			echo "####################################################################"
			nextflow run nextflow/Model_2S.nf -resume -c ~/.nextflow/config -profile $(PROFILE) \
											--PROJECT_NAME $(PROJECT_NAME) --PROJECT_VERSION $(PROJECT_VERSION) \
											--resolution $res --y_interest ${y_interest} \
											--label $(LABEL_FILE)
		done
	done

model_1S: $(LABEL_FILE)
	for y_interest in $(Y_TARGETS)
	do
		for res in $(RESOLUTIONS)
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
			nextflow run nextflow/Model_nn.nf -resume -c ~/.nextflow/config -profile $(PROFILE) \
											--PROJECT_NAME $(PROJECT_NAME) --PROJECT_VERSION $(VERSION) \
											--resolution $res --y_interest ${y_interest} \
											--label $(LABEL_FILE) \
											--input_tiles ./outputs/$(PROJECT_NAME)_$(PROJECT_VERSION)/tiling/${res}/mat_pca/ \
											--mean ./outputs/$(PROJECT_NAME)_$(PROJECT_VERSION)/tiling/${res}/pca_mean/mean.npy
		done
	done

repeated_experiment:
	nextflow run nextflow/Repeated_experiment.nf -resume

necv_vs_ncv:
	nextflow run nextflow/NECV_VS_NCV.nf -resume