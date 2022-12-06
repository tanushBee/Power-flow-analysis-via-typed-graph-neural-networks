import argparse
import logging
import os
import json
import time

import numpy as np
import tensorflow as tf
print(tf.__version__)

# TOPOLOGY CHANGES
from model_PF import TGN_PF

# Build parser
parser = argparse.ArgumentParser()

# Define mode
parser.add_argument('--infer_data', type=str,
    help='If specified, data on which to evaluate a reloaded model. If specified, you should also specify'\
    +' a result_dir!')

# Define training parameters
parser.add_argument('--rdm_seed', type=int,
    help='Random seed. Random by default.')
# default=1000000
parser.add_argument('--max_iter', type=int, default=5000,
    help='Number of training steps')
parser.add_argument('--minibatch_size', type=int, default=20,
    help='Size of each minibatch')
parser.add_argument('--learning_rate', type=float, default=1e-4,
    help='Learning rate')
parser.add_argument('--glob_norm', type=float, default=1,
    help='Gradient global norm')
parser.add_argument('--track_validation', type=float, default=20,
    help='Tracking validation metrics every XX iterations')
parser.add_argument('--data_directory', type=str, default='data/',
    help='Path to the folder containing data')

# Define model parameters
parser.add_argument('--tgn_layers', type=int, default=15,
    help='inference layers')
parser.add_argument('--time_steps', type=int, default=2,
    help='message passing steps')
parser.add_argument('--dim_edges', type=int, default=16,
    help='Edge embedded dimension')
parser.add_argument('--dim_pv', type=int, default=16,
    help='PV node embedded dimension')
parser.add_argument('--dim_pq', type=int, default=16,
    help='PQ node embedded dimension')

# Define directory to store results and models, or to reload from it
parser.add_argument('--result_dir', 
                help='Path to the folder containing model results')

# Setup session
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.set_soft_device_placement(True)
tf.debugging.set_log_device_placement(False)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

if __name__ == '__main__':

    # Get arguments
    args = parser.parse_args()

    # Set tensorflow random seed for reproductibility, if defined
    if args.rdm_seed is not None:
        tf.random.set_seed(args.rdm_seed)
        np.random.seed(args.rdm_seed)

    # Setup results directory
    if args.result_dir is None:
        result_dir = 'results/' + str(int(time.time()))
    else:
        result_dir = args.result_dir

    # Make result directory if it does not exist
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # set logger
    logFile = os.path.join(result_dir, 'model.log')
    logging.basicConfig(
        level=logging.DEBUG,
        filename=logFile,
        format='%(asctime)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M')

    model = TGN_PF(
        lr=args.learning_rate,
        batch_size=args.minibatch_size,
        dim_e=args.dim_edges,
        dim_pv=args.dim_pv,
        dim_pq=args.dim_pq,
        time_steps=args.time_steps,
        tgn_layers=args.tgn_layers,
        directory=result_dir,
        model_to_restore=args.result_dir
    )

    train = False
    if train:
        model.train(max_iter=args.max_iter,
                    glob_norm=args.glob_norm, 
                    save_step=args.track_validation)
    else:
        model.test()
