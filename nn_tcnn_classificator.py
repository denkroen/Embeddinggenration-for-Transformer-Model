import torch.nn.functional as F
from torch import nn, optim


class ClassificationHead(nn.Module):
    def __init__(self, latent_size, num_classes, neuron_config=[256,256]):
        super().__init__()

        self.fc1 = nn.Linear(latent_size, neuron_config[0])
        self.fc2 = nn.Linear(neuron_config[0], neuron_config[1])
        self.fc3 = nn.Linear(neuron_config[1], num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


