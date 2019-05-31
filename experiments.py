# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Functions to compute realism score and StyleGAN truncation sweep."""

import numpy as np
import os
import PIL.Image
from time import time

import dnnlib
from precision_recall import DistanceBlock
from precision_recall import knn_precision_recall_features
from precision_recall import ManifoldEstimator
from utils import initialize_feature_extractor
from utils import initialize_stylegan

#----------------------------------------------------------------------------
# Helper functions.

def save_image(img_t, filename):
    t = img_t.transpose([1, 2, 0])  # [RGB, H, W] -> [H, W, RGB]
    PIL.Image.fromarray(t.astype(np.uint8), 'RGB').save(filename)

def generate_single_image(Gs, latent, truncation, fmt):
    gen_image = Gs.run(latent, None, truncation_psi=truncation, truncation_cutoff=18, randomize_noise=True, output_transform=fmt)
    gen_image = np.clip(gen_image, 0, 255).astype(np.uint8)
    return gen_image

#----------------------------------------------------------------------------

def compute_stylegan_realism(config):
    print('Running StyleGAN realism...')
    rnd = np.random.RandomState(config.random_seed)
    fmt = dict(func=dnnlib.tflib.convert_images_to_uint8)
    num_images = config.num_images
    num_gen_images = config.num_gen_images  # Low and high realism images are selected from these images.
    minibatch_size = config.minibatch_size    
    truncation = config.truncation

    # Get TFRecord datareader object.
    datareader = config.datareader

    # Initialize VGG-16.
    feature_net = initialize_feature_extractor()

    # Initialize StyleGAN generator.
    Gs = initialize_stylegan()

    # Read real images.
    print('Reading real images...')
    real_features = np.zeros([num_images, feature_net.output_shape[1]], dtype=np.float32)
    for begin in range(0, num_images, minibatch_size):
        end = min(begin + minibatch_size, num_images)
        real_batch, _ = datareader.get_minibatch_np(end - begin)
        real_features[begin:end] = feature_net.run(real_batch, num_gpus=config.num_gpus, assume_frozen=True)

    # Estimate manifold of real images.
    print('Estimating manifold of real images...')
    distance_block = DistanceBlock(feature_net.output_shape[1], config.num_gpus)
    real_manifold = ManifoldEstimator(distance_block, real_features, clamp_to_percentile=50)

    # Generate images.
    print('Generating images...')
    latents = np.zeros([num_gen_images, Gs.input_shape[1]], dtype=np.float32)
    fake_features = np.zeros([num_gen_images, feature_net.output_shape[1]], dtype=np.float32)
    for begin in range(0, num_gen_images, minibatch_size):
        end = min(begin + minibatch_size, num_gen_images)
        latent_batch = rnd.randn(end - begin, *Gs.input_shape[1:])
        gen_images = Gs.run(latent_batch, None, truncation_psi=truncation, truncation_cutoff=18, randomize_noise=True, output_transform=fmt)
        fake_features[begin:end] = feature_net.run(gen_images, num_gpus=config.num_gpus, assume_frozen=True)
        latents[begin:end] = latent_batch

    # Estimate quality of individual samples.
    _, realism_scores = real_manifold.evaluate(fake_features, return_realism=True)

    if config.save_images:
        result_dir = os.path.join(config.save_path, 'stylegan_realism', 'truncation%0.2f' % truncation)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

    # Save samples with lowest and highest realism.
    num_saved = config.show_n_images

    # Sort realism scores.
    highest_realism_idx = realism_scores.argsort()[-num_saved:][::-1]
    lowest_realism_idx = realism_scores.argsort()[:num_saved]

    print('Saving %i low and high quality samples...' % num_saved)
    for i in range(num_saved):
        low_idx = lowest_realism_idx[i]
        high_idx = highest_realism_idx[i]

        # Get corresponding latents.
        low_quality_latent = latents[low_idx]
        high_quality_latent = latents[high_idx]

        # Generate images.
        low_quality_img = generate_single_image(Gs, low_quality_latent[None, :], truncation, fmt)[0]
        high_quality_img = generate_single_image(Gs, high_quality_latent[None, :], truncation, fmt)[0]

        if config.save_images:
            low_realism_score = realism_scores[low_idx]
            high_realism_score = realism_scores[high_idx]
            save_image(low_quality_img, os.path.join(result_dir, 'low_realism_%f_%i.png' % (low_realism_score, i)))
            save_image(high_quality_img, os.path.join(result_dir, 'high_realism_%f_%i.png' % (high_realism_score, i)))
        else:
            low_quality_img.show()
            high_quality_img.show()

    print('Done evaluating StyleGAN realism.\n')

#----------------------------------------------------------------------------

def compute_stylegan_truncation(config):
    print('Running StyleGAN truncation sweep...')
    rnd = np.random.RandomState(config.random_seed)
    fmt = dict(func=dnnlib.tflib.convert_images_to_uint8)
    num_images = config.num_images
    minibatch_size = config.minibatch_size

    # Get TFRecord datareader object.
    datareader = config.datareader

    # Initialize VGG-16.
    feature_net = initialize_feature_extractor()

    # Initialize StyleGAN generator.
    Gs = initialize_stylegan()

    metric_results = np.zeros([len(config.truncations), 3], dtype=np.float32)
    for i, truncation in enumerate(config.truncations):
        print('Truncation %g' % truncation)
        it_start = time()

        # Calculate VGG-16 features for real images.
        print('Reading real images...')
        ref_features = np.zeros([config.num_images, feature_net.output_shape[1]], dtype=np.float32)
        for begin in range(0, num_images, minibatch_size):
            end = min(begin + minibatch_size, num_images)
            real_batch, _ = datareader.get_minibatch_np(end - begin)
            ref_features[begin:end] = feature_net.run(real_batch, num_gpus=config.num_gpus, assume_frozen=True)

        # Calculate VGG-16 features for generated images.
        print('Generating images...')
        eval_features = np.zeros([config.num_images, feature_net.output_shape[1]], dtype=np.float32)
        for begin in range(0, num_images, minibatch_size):
            end = min(begin + minibatch_size, num_images)
            latent_batch = rnd.randn(end - begin, *Gs.input_shape[1:])
            gen_images = Gs.run(latent_batch, None, truncation_psi=truncation, truncation_cutoff=18, randomize_noise=True, output_transform=fmt)
            eval_features[begin:end] = feature_net.run(gen_images, num_gpus=config.num_gpus, assume_frozen=True)

        # Calculate k-NN precision and recall.
        state = knn_precision_recall_features(ref_features, eval_features, num_gpus=config.num_gpus)

        # Store results.
        metric_results[i, 0] = truncation
        metric_results[i, 1] = state['precision'][0]
        metric_results[i, 2] = state['recall'][0]

        # Print progress.
        print('Precision: %0.3f' % state['precision'][0])
        print('Recall: %0.3f' % state['recall'][0])
        print('Iteration time: %gs\n' % (time() - it_start))

    # Save results.
    if config.save_txt:
        result_path = config.save_path
        result_file = os.path.join(result_path, 'stylegan_truncation.txt')
        header = 'truncation_psi,precision,recall'
        np.savetxt(result_file, metric_results, header=header,
                   delimiter=',', comments='')

#----------------------------------------------------------------------------
