from torch import nn, optim



class none_wrapper(nn.Module):
    def __init__(self, sensor_channels, window_length):
        super().__init__()
        self.embedding_size = window_length


    def forward(self, x):
        reconst = 0 # needed for an easy train loop
        pred = 0
        embedding = x 

        return embedding, pred, reconst
    
    def get_embedding_size(self):
        return self.embedding_size