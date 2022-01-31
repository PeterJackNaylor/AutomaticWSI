
// STD_MAX = 21
STD_MAX = 5
// REPEATS = 40
REPEATS = 2
main = file("python/repeated_exp/main.py")

process std {
    // conda '/home/pnaylor/miniconda3/envs/bncv/'
    publishDir "results/necv_vs_ncv/files", pattern: "*.csv", overwrite: true
    maxForks 8
    input:
        each std from 1..STD_MAX
        each repeat from 1..REPEATS

    output:
        file "*.csv" into csv_files
    script:
        """
        python $main ncv_single $std $repeat
        """

}

process come_together {

    input:
        file _ from csv_files.collect()

    output:
        file 'final_table.csv' into agg_table

    """
    #!/usr/bin/python
    
    import pandas as pd
    from glob import glob
    
    f = glob('*.csv')
    final_table = pd.concat([pd.read_csv(fi) for fi in f], axis=0)
    final_table.to_csv('final_table.csv')
    """

}

plot = file("python/repeated_exp/necv_plot.py")
process plot {
    publishDir "results/necv_vs_ncv/plot", mode: 'symlink'

    input:
        file table from agg_table
    output:
        file("necv_vs_cv_plot.html")
    script:
        """
        python $plot $table necv_vs_cv_plot.html
        """
}