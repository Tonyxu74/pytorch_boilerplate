from torch import nn
from myargs import args
import torch


class NewModel(nn.Module):
    """Create a custom model in PyTorch"""

    def __init__(self):
        """
        Initializes the model
        """

        super(NewModel, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        """
        Propogate inputs through the model
        :param x: input
        :return: output after passing through model
        """

        x = self.fc(x)
        return x