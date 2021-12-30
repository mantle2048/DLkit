import numpy as np
import torch
class BasePolicy(object):
    def __init__(self, **kwargs):
        super(BasePolicy, self).__init__(**kwargs)

    def update(self, data: torch.Tensor):
        """Return a dictionary of logging information."""
        raise NotImplementedError

    def save(self, filepath: str):
        raise NotImplementedError

    def test(self):
        """Return a dictionary of test logging information."""
        raise NotImplementedError
