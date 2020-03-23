
var=auc
FILE=/mnt/data3/pnaylor/AutomaticWSI/python/nn/get_results.py

for res in 0 1 2
do 
    for y_interest in Residual Prognostic
    do
        for model in model_1S_a model_1S_b model_1S_c model_1S_d owkin weldon_plus_a weldon_plus_b weldon_plus_c conan_a conan_b conan_c
        do
            PATH=/mnt/data3/pnaylor/AutomaticWSI/outputs/nature_3-0/Model_NN_R${res}/${model}/${y_interest}/results

            NAME=/mnt/data3/pnaylor/AutomaticWSI/all_results/${model}_for_${y_interest}_at_res_${res}__

            /cbio/donnees/pnaylor/applications_slurm/miniconda3/bin/python $FILE --path $PATH --name $NAME --variable_to_report $var

        done
    done
done

