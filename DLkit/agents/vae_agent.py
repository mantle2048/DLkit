import os
from typing import Dict
from .base_agent import BaseAgent
import torch
import DLkit.infrastructure.pytorch_utils as ptu
from DLkit.policies.vae_policy import VAEPolicy
from DLkit.user_config import DEFAULT_IMG_DIR
from collections import OrderedDict
from torchvision.utils import save_image


class VAEAgent(BaseAgent):

    def  __init__(self, agent_config: Dict):

        self.agent_config = agent_config
        self.itr = 0

        self.vae = VAEPolicy(
            agent_config['img_size'],
            agent_config['variable_size'],
            agent_config['hidden_sizes'],
            agent_config['learning_rate']
        )

    def train(self, data: torch.Tensor):

        loss = OrderedDict()
        loss['reconstruct_loss'], loss['kl_divergence'] = self.vae.update(data)
        return loss

    def save(self, filepath: str):
        '''
        test and save decoder img during training
        '''
        img_path = os.path.join(filepath, 'img')
        if not os.path.exists(img_path):
            os.mkdir(img_path)

        model_path = os.path.join(filepath, 'model')
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        self.itr += 1
        self.save_image(img_path, self.itr)
        self.save_model(model_path)

    def save_image(self, img_path: str, itr=None):

        img_name = 'decoder' if itr is None else f'decoder{itr}'
        with torch.no_grad():
            z = torch.randn(64, self.agent_config['variable_size']).to(ptu.device)
            sample = self.vae.decoder(z).to(ptu.device)
            save_image(
                sample.view(64, *self.agent_config['img_size']),
                f'{img_path}/{img_name}' + '.png'
            )

    def save_model(self, model_path: str, itr=None):

        self.vae.save(model_path, itr)
