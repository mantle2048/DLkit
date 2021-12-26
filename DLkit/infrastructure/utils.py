from typing import Union, Any, List, Dict, Tuple
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.distributions import Normal, MultivariateNormal
import torch
from collections import OrderedDict

Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}

def get_dataset(
    dataset_name: str,
    dataset_kwargs: Dict
):

    if dataset_name == 'mnist':
        return get_minist_dataset(
            dataset_kwargs['data_dir'],
            dataset_kwargs['batch_size'],
        )

def get_minist_dataset(
    data_dir: str='dataset',
    batch_size: int=64,
) -> Tuple[DataLoader, DataLoader]:
    ''' download MNIST
        params: config Exp config from args
        return: train_Dataloader test_DataLoader '''

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # mean and var for MNIST dataset.
    ])
    data_train = datasets.MNIST(
        root=data_dir,
        train=True,
        transform=transform,
        download=True
    )
    data_test = datasets.MNIST(
        root=data_dir,
        train=False,
        transform=transform,
    )

    train_loader =  DataLoader(
        dataset=data_train,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader =  DataLoader(
        dataset=data_test,
        batch_size=batch_size,
        shuffle=True
    )
    return train_loader, test_loader


def build_mlp(
    input_size: int,
    output_size: Union[int, type(None)],
    hidden_sizes: List[int],
    activation: Activation = 'relu',
    output_activation: Activation = 'identity'
) -> nn.Sequential:
    ''' build net for vae'''

    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    layers = []
    cur_size = input_size
    for size in hidden_sizes:
        layers.append(nn.Linear(cur_size, size))
        layers.append(activation)
        cur_size = size
    if output_size:
        layers.append(nn.Linear(cur_size, output_size))
        layers.append(output_activation)
    return nn.Sequential(*layers)
