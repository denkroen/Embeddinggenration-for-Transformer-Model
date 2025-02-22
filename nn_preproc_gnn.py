import torch.nn.functional as F
from adjacency_matrix import get_adjacency_matrix
import torch
from torch import nn, optim
from torch_geometric.nn import GCNConv
from learnable_GCN import GC_Block
import numpy as np


class preproc_GNN(nn.Module):
    def __init__(self, window_length, sensor_channels, num_sensors):
        super().__init__()

       

        self.num_sensors = num_sensors
        self.sensor_channels = sensor_channels
        self.gnn_layers = []
        self.embedding_size = num_sensors
        self.empty_channel = nn.Parameter(torch.zeros((1, window_length)))

        


        for _ in range(num_sensors):
            self.gnn_layers.append(GC_Block(window_length,window_length,0.1,sensor_channels+1))

        self.gnn_layers = nn.ModuleList(self.gnn_layers)




        

    def forward(self, x):


        
        pred = 0
        reconst = 0

        outputs = []

        gnn_inputs = []
        #empty_channel = self.empty_channel.unsqueeze(1).repeat(x.shape[0], 1, 1)
        empty_channel = torch.zeros(x.shape[0],1, x.shape[2], device=x.device)
        #print(empty_channel.shape)
        #print(x.shape)
        for i in range(0, x.shape[1], self.sensor_channels):
            sensor = x[:,i:i + self.sensor_channels, :]
            gnn_input = torch.cat([sensor, empty_channel],dim=1)
            gnn_inputs.append(gnn_input)

        for i in range(0,self.num_sensors):

            input = gnn_inputs[i]
            gnn_layer = self.gnn_layers[i]

            output = gnn_layer(input)
            output = output[:,self.sensor_channels:self.sensor_channels+1, :] # last row
            #print(self.sensor_channels+1)
            #print(output.type)
            if i == 0:
                outputs = output
            else:
                outputs = torch.cat([outputs, output],dim=1)
            #print(outputs.type)





        

        embedding = outputs


        return embedding, pred, reconst
    
    def get_embedding_size(self):
        return self.embedding_size


    def split_and_append_empty_row(self,x, sensor_channels):
        gnn_inputs = []
        for i in range(0, x.shape[0], sensor_channels):
            sensor = x[i:i + sensor_channels, :]
            empty_channel = torch.zeros((1, x.shape[1]))
            gnn_input = torch.cat([sensor, empty_channel])
            gnn_inputs.append(gnn_input)
        return gnn_inputs


