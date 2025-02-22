import torch.nn.functional as F
from torch import nn, optim


class MLP(nn.Module):
    def __init__(self, sensor_channels, window_length, num_layers, embedding_size):
        super().__init__()
        self.embedding_size = window_length

        dim_in = sensor_channels
        num_neurons = embedding_size

        self.mlp_embedding_generator = nn.ModuleList()

        fc_input = nn.Sequential(nn.Linear(dim_in, num_neurons), nn.ReLU())

        self.mlp_embedding_generator.append(fc_input)

        for _ in range(num_layers-1):
            embed_layer = nn.Sequential(nn.Linear(num_neurons, num_neurons), nn.ReLU())
            self.mlp_embedding_generator.append(embed_layer)


    def forward(self, x):
        pred = 0
        reconst = 0

        #(Batch, 1, sensors, time) -> (Batch,1,Time,sensors)
        #x = x.permute(0,1,3,2) # nedded to apply fc-layer for every timestep over all sensors

        for mod in self.mlp_embedding_generator:
            x = mod(x)
        embedding = x

        #(Batch, 1,time, sensors) -> (Batch,1,sensors,Time)
        #embedding = embedding.permute(0,1,3,2) # change channels back for further processing 
        #:print(embedding.shape)

        return embedding, pred, reconst
    
    def get_embedding_size(self):
        return self.embedding_size
