


STD_MAX = 5
DATA_REPEATS = 2
REPEATS_PER_PROCESS = 2
REPEATS_BY = 1

// STD_MAX = 21
// DATA_REPEATS = 10
// REPEATS_PER_PROCESS = 10
// REPEATS_BY = 100

// ENV = '/home/pnaylor/apps/miniconda3/envs/tmi'
// ENV = '/home/pnaylor/miniconda3/envs/tmi/'
generate_data = file("python/repeated_exp/generate_data.py")

process gene_data {
    // conda "$ENV"

    input:
        each std from 1..STD_MAX
        each repeat from 1..DATA_REPEATS

    output:
        set val("std=${std}--rep=${repeat}"), file("Xy_train.npz"), file("Xy_val.npz"), file("Xy_test.npz") into XY

    script:
        """
        python $generate_data $std
        """
}

train = file("python/repeated_exp/train.py")
early_stopping = [0, 1]
process train {
    // conda "$ENV"

    input:
        set param, file(xy_train), file(xy_val), file(xy_test) from XY
        each repeat from 1..REPEATS_PER_PROCESS
        each val from early_stopping 
    output:
        file "*.csv" into csv_files

    script:
        """
        python $train $param $repeat $REPEATS_BY $val
        """

}

csv_files.collect().collectFile(skip: 1, keepHeader: true)
    .set { ALL_CSV }

plot = file("python/repeated_exp/variance_plot.py")
process come_together {
    publishDir "results/repeated_exp", mode: 'symlink'
    input:
        file out from ALL_CSV

    output:
        set file("${out}"), file("variance_mean_plot.html"), file("only_variance.html")

    """
    python $plot $out variance_mean_plot.html only_variance.html
    """
}