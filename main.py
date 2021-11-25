import argparse
import logging

from agent.utils import train_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # Sub-command for training a policy
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--config', type=str, required=True)
    train_parser.add_argument('--algo', type=str, required=True)
    train_parser.add_argument('--model_dir', type=str, required=True)
    train_parser.add_argument('--load_dir', type=str)

    train_parser.add_argument('--timestep', type=str)
    train_parser.add_argument('-sh', '--shaped', action='store_true')
    train_parser.add_argument('-s', '--simple', action='store_true')
    train_parser.add_argument('-v', '--visualize', action='store_true')
    train_parser.add_argument('-tf', '--timefeature', action='store_true')

    train_parser.set_defaults(func=train_utils.train)

    # Sub-command for running a trained policy
    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('--model', type=str)
    run_parser.add_argument('--config', type=str, required=True)
    run_parser.add_argument('-v', '--visualize', action='store_true')
    run_parser.add_argument('-t', '--test', action='store_true')
    run_parser.add_argument('-s', '--stochastic', action='store_true')

    run_parser.set_defaults(func=train_utils.run)

    logging.getLogger().setLevel(logging.DEBUG)

    args = parser.parse_args()
    args.func(args)