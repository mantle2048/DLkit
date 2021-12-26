from typing import Dict
class BaseAgent(object):

    def __init__(self, **kwargs):
        super(BaseAgent, self).__init__(**kwargs)

    def train(self) -> Dict:
        """Return a dictionary of logging information."""
        raise NotImplementedError

    def save(self, filepath: str):
        raise NotImplementedError



