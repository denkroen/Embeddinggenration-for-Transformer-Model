import torch.nn.functional as F
from adjacency_matrix import get_adjacency_matrix
import torch
from torch import nn, optim
from torch_geometric.nn import GCNConv, Sequential
from learnable_GCN import GC_Block
from nn_preproc_gnn import preproc_GNN


class GNN(nn.Module):
    def __init__(self, window_length, embedding_size, num_layers, sensor_channels, num_sensors, learnable_adj):
        super().__init__()

       
        self.gls = True
        self.learnable_adj = learnable_adj
        if self.gls == True:

            if learnable_adj == True:
                self.preproc_gnn = True ## turn False if no preproc
                self.preproc = preproc_GNN(window_length,sensor_channels,num_sensors)
                num_nodes = num_sensors
                self.embedding_size = embedding_size
                self.adj_matrix = nn.Parameter(torch.FloatTensor(num_sensors, num_sensors))
                #self.adj_matrix = nn.Parameter(torch.ones(num_nodes, num_nodes))
            else:
                self.preproc_gnn = True
                self.embedding_size = embedding_size
                num_nodes = num_sensors
                self.preproc = preproc_GNN(window_length,sensor_channels,num_sensors)
                self.adj_matrix = get_adjacency_matrix(num_sensors) 

        else: 
            self.preproc_gnn = False
            num_nodes = num_sensors*sensor_channels
            self.embedding_size = embedding_size





        #self.gcbs = []
    #for i in range(num_stage):
    #  self.gcbs.append(GC_Block(
    #      self._hidden_dim, 
    #      p_dropout=p_dropout, 
    #      output_nodes=output_nodes)
    #  )

    #self.gcbs = nn.ModuleList(self.gcbs)


        
        self.gnn_embedding_generator = []

        #gcn_input = Sequential('x,adj,index',[(GCNConv(window_length, embedding_size),'x,adj,index -> x'),nn.ReLU(inplace=True)])
        gcn_input = GC_Block(window_length,embedding_size,0.1,num_nodes)


        self.gnn_embedding_generator.append(gcn_input)

        for _ in range(num_layers-1):
            self.gnn_embedding_generator.append(GC_Block(embedding_size,embedding_size,0.1,num_nodes))
 
        self.gnn_embedding_generator = nn.ModuleList(self.gnn_embedding_generator)

        

    def forward(self, x):


        
        pred = 0
        reconst = 0
        #bring att to device
        #att = self.att.to(x.device)
        #att = F.tanh(att)

        x = torch.squeeze(x,1) # important for the lara data. (Batch, 1, sensors, time) -> (Batch,sensors,time)
        x = x.permute(0, 2, 1)

        if self.preproc_gnn:
            x, _, _ = self.preproc.forward(x)
            #print(x.type)
            #adj_matrix = self.adj_matrix
        #else:

            #adj_matrix = F.sigmoid(F.relu(self.adj_matrix))
            #edge_index = (adj_matrix > 0.5).nonzero().t()
            #edge_weight = adj_matrix[edge_index[0], edge_index[1]]


        

        for mod in self.gnn_embedding_generator:
            if self.preproc_gnn:
                adj_matrix = self.adj_matrix.to(dtype=torch.float)
                adj_matrix = adj_matrix.to(x.device)
                if self.learnable_adj:
                    adj_matrix = F.tanh(adj_matrix)

                x = mod(x,adj_matrix)
            else:
                x = mod(x) 

        x = x.permute(0, 2, 1)


        embedding = x
        #print(embedding.shape)


        return embedding, pred, reconst
    
    def get_embedding_size(self):
        return self.embedding_size


