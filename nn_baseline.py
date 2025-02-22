from torch import nn, optim
import torch




class baseline_CNN(nn.Module):
    """
    Architecture based on : https://github.com/yolish/har-with-imu-transformer/blob/main/models/IMUTransformerEncoder.py
    """
    def __init__(self, window_length,sensor_channels, embedding_size):
        super().__init__()
        self.embedding_size = window_length
        num_neurons = embedding_size
    
        self.convolutions = nn.Sequential(nn.Conv1d(sensor_channels, num_neurons, 1), nn.GELU(),
                                        nn.Conv1d(num_neurons, num_neurons, 1), nn.GELU(),
                                        nn.Conv1d(num_neurons, num_neurons, 1), nn.GELU(),
                                        nn.Conv1d(num_neurons, num_neurons, 1), nn.GELU())


      


    def forward(self, x):
        reconst = 0 # needed for an easy train loop b w d
        pred = 0

        x = torch.squeeze(x,1)

        x = self.convolutions.forward(x.transpose(1,2))
        embedding = x.permute(0,2,1) 

        return embedding, pred, reconst
    
    def get_embedding_size(self):
        return self.embedding_size