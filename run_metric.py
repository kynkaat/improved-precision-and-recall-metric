# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Script to run StyleGAN truncation sweep or evaluate realism score of StyleGAN samples."""

import argparse
import os
import tensorflow as tf

import dnnlib
from dnnlib.util import Logger
from ffhq_datareader import load_dataset
from experiments import compute_stylegan_realism
from experiments import compute_stylegan_truncation
from utils import init_tf

SAVE_PATH = os.path.dirname(__file__)

#----------------------------------------------------------------------------
# Configs for truncation sweep and realism score.

realism_config = dnnlib.EasyDict(minibatch_size=8, num_images=50000, num_gen_images=1000, show_n_images=64,
                                 truncation=1.0, save_images=True, save_path=SAVE_PATH, num_gpus=1,
                                 random_seed=123456)

truncation_config = dnnlib.EasyDict(minibatch_size=8, num_images=50000, truncations=[1.0, 0.7, 0.3],
                                    save_txt=True, save_path=SAVE_PATH, num_gpus=1, random_seed=1234)

#----------------------------------------------------------------------------
# Minimal CLI.

def parse_command_line_arguments(args=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Improved Precision and Recall Metric for Assessing Generative Models.',
                                     epilog='This script can be used to reproduce StyleGAN truncation sweep (Fig. 4) and' \
                                            ' computing realism score for StyleGAN samples (Fig. 11).')

    parser.add_argument(
        '-d',
        '--data_dir',
        type=str,
        required=True,
        help='Absolute path to TFRecords directory.'
    )
    parser.add_argument(
        '-t',
        '--truncation_sweep',
        action='store_true',
        help='Calculate StyleGAN truncation sweep. Replicates Fig. 4 from the paper.'
    )
    parser.add_argument(
        '-r',
        '--realism_score',
        action='store_true',
        help='Calculate realism score for StyleGAN samples. Replicates Fig. 11 from Appendix.'
    )
    parsed_args, _ = parser.parse_known_args(args)
    return parsed_args

#----------------------------------------------------------------------------

def main(args=None):
    # Parse command line arguments.
    parsed_args = parse_command_line_arguments(args)

    # Initialize logger.
    Logger()

    # Initialize dataset object.
    init_tf()
    dataset_obj = load_dataset(tfrecord_dir=parsed_args.data_dir, repeat=True, shuffle_mb=0,
                               prefetch_mb=100, max_label_size='full', verbose=True)

    if parsed_args.realism_score:  # Compute realism score.
        realism_config.datareader = dataset_obj
        compute_stylegan_realism(**realism_config)

    if parsed_args.truncation_sweep:  # Compute truncation sweep.
        truncation_config.datareader = dataset_obj
        compute_stylegan_truncation(**truncation_config)

    peak_gpu_mem_op = tf.contrib.memory_stats.MaxBytesInUse()
    peak_gpu_mem_usage = peak_gpu_mem_op.eval()
    print('Peak GPU memory usage: %g GB' % (peak_gpu_mem_usage * 1e-9))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
