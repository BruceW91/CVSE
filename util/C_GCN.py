import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from util.util_C_GCN import *


def l2norm(X, dim=-1, eps=1e-12):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, which shared the weight between two separate graphs
    """
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, conv_mode='whole_graph'):
        '''
        Graph Conv function
        :param input: input signal
        :param adj: adj graph dict [OPC, OMC, all]
        :param conv_mode: choose which graph to make convolution (separate graphs or whole graph)
        '''

        if conv_mode=='dual_graph':
            support = torch.matmul(input, self.weight)

            output_1 = torch.matmul(adj['adj_O_P'], support)
            output_2 = torch.matmul(adj['adj_O_M'], support)
            output = (output_1 + output_2) / 2

            if self.bias is not None:
                return output + self.bias
            else:
                return output

        elif conv_mode=='whole_graph':
            support = torch.matmul(input, self.weight)
            output = torch.matmul(adj['adj_all'], support)
            if self.bias is not None:
                return output + self.bias
            else:
                return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class C_GCN(nn.Module):

    def __init__(self, num_classes, in_channel=300, t=0, adj_file=None, norm_func='sigmoid', adj_gen_mode='C_GCN', opt=None):
        super(C_GCN, self).__init__()

        self.num_classes = num_classes
        self.gc1 = GraphConvolution(in_channel, opt.embed_size // 2)
        self.gc2 = GraphConvolution(opt.embed_size // 2,  opt.embed_size)
        self.relu = nn.LeakyReLU(0.2)

        # concept correlation mat generation
        _adj = gen_A_concept(num_classes, t, adj_file, gen_mode=adj_gen_mode)

        self.adj_O_P = Parameter(torch.from_numpy(_adj['adj_O_P']).float())
        self.adj_O_M = Parameter(torch.from_numpy(_adj['adj_O_M']).float())
        self.adj_all = Parameter(torch.from_numpy(_adj['adj_all']).float())

        self.norm_func = norm_func
        self.softmax = nn.Softmax(dim=1)
        self.joint_att_emb = nn.Linear(opt.embed_size, opt.embed_size)
        self.embed_size = opt.embed_size
        self.init_weights()

    def init_weights(self):
        """Xavier initialization"""
        r = np.sqrt(6.) / np.sqrt(self.embed_size + self.embed_size)
        self.joint_att_emb.weight.data.uniform_(-r, r)
        self.joint_att_emb.bias.data.fill_(0)


    def forward(self, feature, inp, conv_mode='whole_graph'):

        inp = inp[0]

        adj_O_P = gen_adj(self.adj_O_P ).detach()
        adj_O_M = gen_adj(self.adj_O_M ).detach()
        adj_all = gen_adj(self.adj_all).detach()

        adj = {}
        adj['adj_O_P'] = adj_O_P
        adj['adj_O_M'] = adj_O_M
        adj['adj_all'] = adj_all

        x = self.gc1(inp, adj, conv_mode=conv_mode)
        x = self.relu(x)
        x = self.gc2(x, adj, conv_mode=conv_mode)

        concept_feature = x
        concept_feature = l2norm(concept_feature)

        return concept_feature


    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]




