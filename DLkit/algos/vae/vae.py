import time
import argparse
import numpy as np
import torch
from torch import nn
from typing import Dict, Tuple, Union, List, Any
from torch import optim
from collections import OrderedDict

import DLkit.infrastructure.pytorch_utils as ptu
from DLkit.agents.vae_agent import VAEAgent
from DLkit.infrastructure.dl_trainer import DLTrainer

# use VAE to autoencoder MNIST dataset
# assume two hidden variable

class VAETrainer():

    def __init__(self, config: Dict):

        # build data_loader and vae
        agent_config = {
                'img_size': (1, 28, 28),
                'variable_size': 4,
                'hidden_sizes': [512,256],
                'learning_rate': config['learning_rate'],
                }

        self.config = config
        self.config['agent_class'] = VAEAgent
        self.config['agent_config'] = agent_config
        self.dl_trainer = DLTrainer(self.config)

    def run_training_loop(self):

        self.dl_trainer.run_training_loop(
                epoches = self.config['epoches'],
                train_loader = self.dl_trainer.train_loader,
                test_loader = self.dl_trainer.test_loader
                )

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='vae_params')

    parser.add_argument('--exp_name', type=str, default='vae_mnist')
    parser.add_argument('--dataset_name', type=str, default='mnist')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--log_dir', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epoches', type=int, default=20)
    parser.add_argument('--which_gpu', type=int, default=0)
    parser.add_argument('--use_gpu', action='store_true', default=False)
    parser.add_argument('--datestamp', action='store_true', default=False)
    args = parser.parse_args()

    config = vars(args)

    trainer = VAETrainer(config)
    trainer.run_training_loop()

