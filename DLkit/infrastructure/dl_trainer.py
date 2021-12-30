import os
import time
import torch
import numpy as np
import pandas as pd
import DLkit.infrastructure.pytorch_utils as ptu
import seaborn as sns

from DLkit.infrastructure.logx import EpochLogger, setup_logger_kwargs
from DLkit.infrastructure.utils import get_dataset
from DLkit.agents.vae_agent import VAEAgent
from DLkit.infrastructure.plot import plot_data
from matplotlib import pyplot as plt
from typing import Dict, Tuple, List, Any, Union
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

            for data, label in test_loader:
                test_log = self.agent.test(data)
                self.logger.store(**test_log)

            self.logger.log_tabular('Epoch', epoch)
            self.logger.log_tabular('Exp', self.config['exp_name'])
            self.logger.log_tabular('Time', (time.time() - self.start_time) / 60)

            for key, value in self.logger.epoch_dict.items():
                self.logger.log_tabular(key, average_only=True)

            self.logger.dump_tabular()

            self.agent.save(filepath=self.logger.output_dir)


    def plot_data(self, xaxis: str, yaxis: Union[str, list]):

        progress_path = os.path.join(self.logger.output_dir, 'progress.txt')
        data = pd.read_table(progress_path)

        subplot_num = len(yaxis)
        fig, axs = plt.subplots(subplot_num, 1 , figsize=(8, subplot_num * 6))

        if isinstance(yaxis, str): yaxis = [yaxis]

        for value, ax in zip(yaxis, axs):

            sns.set(style='whitegrid', palette='tab10', font_scale=1.5)
            sns.lineplot(data=data, x=xaxis, y=value, ax=ax, linewidth=3.0)
            leg = ax.legend(labels = [value], loc='best') #.set_draggable(True)
            for line in leg.get_lines():
                line.set_linewidth(3.0)
            ax.set_title(value)
            xscale = np.max(np.asarray(data[xaxis])) > 5e3
            if xscale:
                # Just some formatting niceness: x-axis scale in scientific notation if max x is large
                ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

        img_path = os.path.join(self.logger.output_dir, 'img')
        if not os.path.exists(img_path):
            os.mkdir(img_path)
        fig.savefig(fname=os.path.join(img_path, 'progress.png'), dpi=150)

