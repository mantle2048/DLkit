import time
import torch
import numpy as np
import DLkit.infrastructure.pytorch_utils as ptu

from DLkit.infrastructure.logx import EpochLogger, setup_logger_kwargs
from DLkit.infrastructure.utils import get_dataset
from DLkit.agents.vae_agent import VAEAgent
from typing import Dict, Tuple, List, Any
from collections import defaultdict, OrderedDict

class DLTrainer():

    def __init__(self, config: Dict):

        # init config and logger
        self.config = config


        # init Gpu and Seed
        torch.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        ptu.init_gpu(
            self.config['use_gpu'],
            self.config['which_gpu']
        )

        agent_class = self.config['agent_class']
        self.agent = agent_class(self.config['agent_config'])

        # init logger config
        logger_kwargs = setup_logger_kwargs(
                exp_name=self.config['exp_name'],
                seed=self.config['seed'],
                data_dir=self.config['log_dir'],
                datestamp=self.config['datestamp']
                )
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(config)
        self.start_time = time.time()

        # init exp config
        self.epoches = self.config['epoches']



        dataset_kwargs = {
                'data_dir': self.config['data_dir'],
                'batch_size': self.config['batch_size']
                }

        self.train_loader, self.test_loader = get_dataset(self.config['dataset_name'], dataset_kwargs)


    def run_training_loop(self, epoches: int, train_loader, test_loader):

        for epoch in range(epoches):

            for data, label in train_loader:
                train_log = self.agent.train(data)
                self.logger.store(**train_log)

            self.logger.log_tabular('Exp', self.config['exp_name'])
            self.logger.log_tabular('Time', (time.time() - self.start_time) / 60)

            for key, value in self.logger.epoch_dict.items():
                    self.logger.log_tabular(key, average_only=True)

            self.logger.dump_tabular()

            self.agent.save(filepath=self.logger.output_dir)



