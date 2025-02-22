import torch.nn.functional as F
from torch import nn
import torch
from nn_tcnn_autoencoder_encoder import TCNNEncoder
from nn_tcnn_autoencoder_decoder import TCNNDecoder
from nn_tcnn_autoencoder_classificator import AutoencoderClassificationHead

from utils import compute_feature_map_size_tcnn



class TCNN_Autoencoder(nn.Module):
    def __init__(self, in_channels, window_length, sensor_channels, filter_size, num_classes, num_filters):
        super().__init__()
        W,H = compute_feature_map_size_tcnn(0,sensor_channels,window_length,filter_size)

        self.embedding_size = W
        latent_size = W*H*256

        self.encoder = TCNNEncoder(in_channels, num_filters, filter_size)
        self.decoder = TCNNDecoder(in_channels, num_filters, filter_size)
        #self.classificator = AutoencoderClassificationHead(latent_size,num_classes)

        
    def forward(self, x):
        x = self.encoder.forward(x)
        feat_rep = x
        reconst = self.decoder.forward(x)

        #x = x.view(x.size()[0], x.size()[1], x.size()[2])
        #x = torch.flatten(x,1)
        #pred = self.classificator.forward(x)
        pred = 0 #no classification head

        return feat_rep, pred, reconst
    
    def get_embedding_size(self):
        return self.embedding_size