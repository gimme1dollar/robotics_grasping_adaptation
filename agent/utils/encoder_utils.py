import argparse
import os
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from agent.utils import io_utils
from agent.robot import robot, actuator, sensor, encoder

def _load_data_set(data_path, test):
    with open(os.path.expanduser(data_path), 'rb') as f:
        dataset = pickle.load(f)
    return dataset['test'] if test else dataset['train']

def _preprocess_img(data_set):
    result = []

    rgb_imgs = data_set['rgb']
    depth_imgs = data_set['depth']
    masks = data_set['masks']

    for i in range(depth_imgs.shape[0]):
        masks[i][(masks[i] <= 1)] = 0.
        masks[i][(masks[i] > 1)] = 1.

        depth_imgs[i][(masks[i] == 0)] = 0.
        
        tmp = np.dstack((rgb_imgs[i]/255., depth_imgs[i]))
        result.append(tmp)
    result = np.asarray(result)
    return result

def train(args):
    # Load the encoder configuration
    config = io_utils.load_yaml(args.config)

    # If not existing, create the model directory
    model_dir = os.path.expanduser(args.model_dir)
    os.makedirs(model_dir, exist_ok=True)
    io_utils.save_yaml(config, os.path.join(model_dir, 'encoder_config.yaml'))
    
    # Build the model
    if config['encoder_type'] == 'simple':
        model = encoder.SimpleAutoEncoder(config)
    elif config['encoder_type'] == 'domain':
        model = encoder.DomainAdaptingEncoder(config)
        raise NotImplemented
        

    # Load and process the training data
    train_set = _load_data_set(config['data_path'], test=False)
    train_imgs = _preprocess_img(train_set)

    # Train the model
    batch_size = config['batch_size']
    epochs = config['epochs']

    model.train(train_imgs, train_imgs, batch_size, epochs, model_dir)


def test(args):
    # Load the model
    config = io_utils.load_yaml(os.path.join(args.model_dir, 'encoder_config.yaml'))

    if config['encoder_type'] == 'simple':
        model = encoder.SimpleAutoEncoder(config)
    elif config['encoder_type'] == 'domain':
        model = encoder.DomainAdaptingEncoder(config)
    model.load_weights(args.model_dir)

    # Load the test set
    test_set = _load_data_set(config['data_path'], test=True)
    test_imgs = _preprocess_img(test_set)

    # Compute the test loss
    loss = model.test(test_imgs, test_imgs)
    print('Test loss: {}'.format(loss))
    return loss


def visualize(args):
    n_imgs = 5   # number of images to visualize

    # Load the model
    config = io_utils.load_yaml(os.path.join(args.model_dir, 'encoder_config.yaml'))
    model = encoder.SimpleAutoEncoder(config)
    model.load_weights(args.model_dir)

    # Load and process a random selection of test images
    test_set = _load_data_set(config['data_path'], test=False)
    selection = np.random.choice(test_set['rgb'].shape[0], size=n_imgs)
    processed = _preprocess_img(test_set)[selection]
    proc_rgb   = processed[:,:,:,:3]
    proc_depth = processed[:,:,:,-1]

    # Encode/reconstruct images and compute errors
    reconstruction = model.predict(processed)
    recon_rgb   = reconstruction[:,:,:,:3]
    recon_depth = reconstruction[:,:,:,-1]

    error_rgb = np.abs(proc_rgb - recon_rgb)
    error_depth = np.abs(proc_depth - recon_depth)

    # Plot results
    fig = plt.figure()
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(n_imgs, 6),
                     share_all=True,
                     axes_pad=0.05)

    for i in range(n_imgs):
        def _add_rgb_img(img, j):
            ax = grid[6 * i + j]
            ax.set_axis_off()
            ax.imshow(img)

        def _add_depth_img(img, j):
            ax = grid[6 * i + j]
            ax.set_axis_off()
            ax.imshow(img.squeeze(), cmap='gray')

        # Plot depth, reconstruction and error
        _add_rgb_img(proc_rgb[i], 0)
        _add_rgb_img(recon_rgb[i], 1)
        _add_depth_img(error_rgb[i], 2)
        _add_depth_img(proc_depth[i], 3)
        _add_depth_img(recon_depth[i], 4)
        _add_depth_img(error_depth[i], 5)

    plt.savefig(os.path.join(args.model_dir, 'reconstructions.png'), dpi=300)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=str)

    subparsers = parser.add_subparsers()

    # Sub-command for training the model
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--config', type=str, required=True)
    train_parser.set_defaults(func=train)

    # Sub-command for testing the model
    test_parser = subparsers.add_parser('test')
    test_parser.set_defaults(func=test)

    # sub-command for visualizing reconstructed images
    vis_parser = subparsers.add_parser('visualize')
    vis_parser.set_defaults(func=visualize)

    args = parser.parse_args()
    args.func(args)
