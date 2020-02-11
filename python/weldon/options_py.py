
def get_options():
    import argparse
    parser = argparse.ArgumentParser(
        description='dispatches a set of files into folds')
    parser.add_argument('--path', required=True,
                        metavar="str", type=str,
                        help='patient for patients')
    parser.add_argument('--fold_test', required=True,
                        metavar="str", type=int,
                        help='fold number for testing')
    parser.add_argument('--table', required=True,
                        metavar="str", type=str,
                        help='patient label table and fold')
    parser.add_argument('--inner_cross_validation_number', required=True,
                        metavar="int", type=int,
                        help='number of sub folds for inner cross validation')
    parser.add_argument('--batch_size', required=True,
                        metavar="str", type=int,
                        help='batch size')
    parser.add_argument('--epochs', required=True,
                        metavar="str", type=int,
                        help='number of epochs')
    parser.add_argument('--seed', required=True,
                        metavar="str", type=int,
                        help='seed')
    parser.add_argument('--cpus',
                        metavar="str", type=int, default=8,
                        help='number of available cpus')
    parser.add_argument('--main_name', required=False,
                        metavar="str", type=str,
                        help='main name for saving')  
    parser.add_argument('--mean_name', required=True,
                        metavar="str", type=str,
                        help='mean array')  
    parser.add_argument('--size', required=True,
                        metavar="str", type=int,
                        help='number of instances per patient')
    parser.add_argument('--class_type', required=True,
                        metavar="str", type=str,
                        help='classification problem at hand')
    parser.add_argument('--gaussian_noise',
                        metavar="str", type=int, default=0,
                        help='if to apply gaussian noise to the input')
    parser.add_argument('--pool',
                        metavar="str", type=str, default='max',
                        help='if to apply max or average pooling')
    parser.add_argument('--repeat', required=True,
                        metavar="str", type=int,
                        help='number of repeats')
    parser.add_argument('--n_fold',
                        metavar="int", type=int, default=10,
                        help='number of fold')
    parser.add_argument('--y_variable', required=False,
                        metavar="str", type=str,
                        help='name of the variable to predict Residual | Prognostic | fold ...')
    args = parser.parse_args()

    if args.gaussian_noise == 0:
        args.gaussian_noise = False
    else:
        args.gaussian_noise = True

    args.optimizer_name = "Adam"
    args.max_queue_size = 10
    args.workers = 1
    args.use_multiprocessing = False
    args.input_depth = 2048 # 512

    args.learning_rate_start = -6
    args.learning_rate_stop = 1
    args.weight_decay_start = -4
    args.weight_decay_stop = 1

    args.hidden_fcn_list = [32, 64, 128, 256]
    args.hidden_btleneck_list = [4, 8, 32, 64]

    args.pooling_layer = args.pool
    args.k = 5
    args.n_scores = 5
    args.concatenate = False
    args.output_name = "neural_networks_model_fold_number_{}.csv".format(args.fold_test)


    return args
