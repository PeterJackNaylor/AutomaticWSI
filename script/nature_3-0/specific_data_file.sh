
NUMBER_OF_FOLDS=10

# Not to run, as this was run with CellularHeatmaps to generate the labels.
python python/label_standardise/substrat_label.py --input_table ../Data/Biopsy_csv/rcb_substrat_updated_with_gui_and_with_present_cluster.csv \
                                                  --folder_to_check ../Data/Biopsy_Nature_3-0 \
                                                  --output_table ../Data/Biopsy_csv/rcb_substrat_before_check_Nature_3-0.csv
python python/label_standardise/ftnbc_label.py --input_table ../Data/Biopsy_csv/MarickDataSet.xlsx \
                                               --folder_to_check ../Data/Biopsy_Nature_3-0/ \
                                               --output_table ../Data/Biopsy_csv/ftnbc_Nature_3-0.csv
python python/label_standardise/merge_data.py --substra ../Data/Biopsy_csv/rcb_substrat_before_check_Nature_3-0.csv \
                                                     --ftnbc ../Data/Biopsy_csv/ftnbc_Nature_3-0.csv \
                                                     --folder ../Data/Biopsy_Nature_3-0/ \
                                                     --output_name ../Data/Biopsy_csv/label_Nature_3-0.csv
rm -r ../Data/Biopsy_csv/ftnbc_Nature_3-0.csv ../Data/Biopsy_csv/rcb_substrat_before_check_Nature_3-0.csv  ../Data/Biopsy_csv/not_in_table
mkdir outputs
python python/label_standardise/create_test_folds.py --table ../Data/Biopsy_csv/label_Nature_3-0.csv --output_name ./outputs/nature_3-0/label.csv --folds $NUMBER_OF_FOLDS