import os
import torch
import itertools
import numpy as np
from torch import nn
from torch import optim
from .base_policy import BasePolicy
from typing import List, Dict, Tuple

from DLkit.infrastructure.utils import build_mlp
import DLkit.infrastructure.pytorch_utils as ptu

class VAEPolicy(BasePolicy, nn.Module):

    def __init__(
        self,
        img_size: tuple=(1, 28, 28),
        variable_size: int=2,
        hidden_sizes: List=[256,256],
        learing_rate: float=1e-3,
        **kwargs
    ):

        nn.Module.__init__(self)
        self.img_size = img_size
        self.input_size = np.prod(self.img_size)
        self.variable_size = variable_size
        self.hidden_sizes = hidden_sizes
        self
        self.lr = learing_rate

        self.encoder_net = build_mlp(self.input_size, None, self.hidden_sizes)
        self.mu_net = nn.Linear(self.hidden_sizes[-1], self.variable_size)
        self.logstd_net = nn.Linear(self.hidden_sizes[-1], self.variable_size)

        self.decoder_net = build_mlp(self.variable_size, self.input_size, self.hidden_sizes, output_activation='sigmoid')

        self.encoder_net.to(ptu.device)
        self.decoder_net.to(ptu.device)
        self.mu_net.to(ptu.device)
        self.logstd_net.to(ptu.device)

        self.BCELoss = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)

    def encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor]:

        x = self.encoder_net(x)
        mu = self.mu_net(x)
        logstd  = self.logstd_net(x)
        std = torch.exp(logstd)

        return mu, std

    def sample(self, mu: torch.Tensor, std: torch.Tensor):
        ''' Reparameterization Trick '''

        batch_size = mu.shape[0]
        epsilon = torch.randn(batch_size, self.variable_size).to(ptu.device)
        z = mu + torch.mul(epsilon, std)

        return z

    def decoder(self, z: torch.Tensor) -> torch.Tensor:

        x = self.decoder_net(z)

        return x

    def update(self, x: torch.Tensor):
        '''
        p = N(mu1,sigma1), q = N(mu2, sigma2)
        KL(p||q) = log(sigma2/sigma1) + (sigma1^2 + (mu1-mu2)^2) / 2*sigma2^2 - 1/2
        '''

        batch_size = x.shape[0]
        x = x.view(batch_size, -1).to(ptu.device)
        mu, std = self.encoder(x)
        z = self.sample(mu, std)
        reconstruct_x = self.decoder(z)

        reconstruction_loss = self.BCELoss(reconstruct_x, x)
        kl_divergence = 0.5 * (torch.pow(std, 2) + torch.pow(mu, 2) - 1 - 2 * torch.log(std)).mean()


        elbo_loss = reconstruction_loss + kl_divergence
        self.optimizer.zero_grad()
        elbo_loss.backward()
        self.optimizer.step()

        return reconstruction_loss.item(), kl_divergence.item()

    def save(self, model_path: str, itr=None):

        model_name = 'model.pth' if itr is None else f'model{itr}.pth'
        torch.save(self.state_dict(), os.path.join(model_path, model_name))


