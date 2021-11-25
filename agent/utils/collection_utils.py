"""Collect images of the picking task for training the autoencoder."""

import argparse
import os
import pickle

import gym
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from agent.utils import io_utils
from agent.robot import robot, actuator, sensor


def collect_dataset(args):
    """Use a random agent on the simplified task formulation to collect a
    dataset of task relevant pictures.
    """
    # config = io_utils.load_yaml(args.config)
    config = args.config
    data_path = os.path.expanduser(args.data_path)

    # height = config['sensor']['camera_info']['height']
    # width = config['sensor']['camera_info']['width']
    height = 64
    width = 64
    rng = np.random.random.__self__  # pylint: disable=E1101

    def collect_imgs(n_imgs, test):
        # Preallocate memory
        rgb_imgs = np.empty((n_imgs, height, width, 3), dtype=np.uint8)
        depth_imgs = np.empty((n_imgs, height, width, 1), dtype=np.float32)
        masks = np.empty((n_imgs, height, width, 1), dtype=np.int32)

        env = gym.make('robot-env-v0', config=config)
        env.reset()

        actuator = env.get_actuator()
        sensor = env.get_camera()

        i = 0
        while (True):
            if i == n_imgs: 
                break

            # Render and store imgs
            rgb, depth, mask = sensor.get_state()
            rgb_imgs[i] = rgb
            depth_imgs[i, :, :, 0] = depth
            masks[i, :, :, 0] = np.reshape(mask, (64, 64))

            # Move the robot
            actuator.step(actuator.action_space.sample())
            i += 1
            if(i % 1000 == 0): 
                env.reset()
                print(f"collected \t{i} data")

        return rgb_imgs, depth_imgs, masks

    # Collect train images
    rgb, depth, masks = collect_imgs(args.n_train_imgs, test=False)
    train_set = {'rgb': rgb, 'depth': depth, 'masks': masks}

    # Collect test images
    rgb, depth, masks = collect_imgs(args.n_test_imgs, test=True)
    test_set = {'rgb': rgb, 'depth': depth, 'masks': masks}

    # Dump the dataset to disk
    dataset = {'train': train_set, 'test': test_set}
    with open(data_path, 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data_path', type=str, default="./checkpoints/encoder/data.pkl")
    parser.add_argument('--n_train_imgs', type=int, default=50_000)
    parser.add_argument('--n_test_imgs', type=int, default=100)
    args = parser.parse_args()

    collect_dataset(args)