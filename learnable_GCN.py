from torch.nn import Parameter
import torch.nn as nn
import torch
import math
import torch.nn.functional as F



class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=False, node_n=0):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        #mask = (F.sigmoid(self.att) > 0.3).t()
        #mask = mask.to(torch.float32)
        support = torch.matmul(input, self.weight)
        if adj == None:
            output = torch.matmul(F.tanh(self.att), support) ###????
        else:
            output = torch.matmul(adj,support) # if specified adj_matrix
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    

class GC_Block(nn.Module):
  def __init__(self, in_features, out_features, p_dropout, node_n=0, bias=False):

        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.gc1 = GraphConvolution(
            in_features, out_features,
            node_n=node_n, 
            bias=bias
        )
        self.bn1 = nn.BatchNorm1d(node_n * out_features)
        
        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

  def forward(self, x, adj=None):
        """Forward pass of one block (gcn, bn, act_f, dropout)"""
        y = self.gc1(x,adj)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)
        return y

  def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
                + str(self.out_features) + ')'

