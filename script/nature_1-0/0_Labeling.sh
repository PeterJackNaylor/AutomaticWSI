# Not to run, as this was run with CellularHeatmaps to generate the labels.
python python/label_standardise/substrat_label.py -resume --input_table ../Data/Biopsy_csv/rcb_substrat.csv \
                                                  --folder_to_check ../Data/Biopsy_guillaume \
                                                  --output_table ../Data/Biopsy_csv/rcb_substrat_after.csv
python python/label_standardise/ftnbc_label.py -resume --input_table ../Data/Biopsy_csv/MarickDataSet.xlsx \
                                               --folder_to_check ../Data/Biopsy/ \
                                               --output_table ../Data/Biopsy_csv/ftnbc.csv
python python/label_standardise/create_test_folds.py -resume --substra ../Data/Biopsy_csv/rcb_substrat_after.csv \
                                                     --ftnbc ../Data/Biopsy_csv/ftnbc.csv \
                                                     --output_name ./outputs/label.csv
