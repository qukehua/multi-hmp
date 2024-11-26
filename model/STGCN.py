#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math


class ConvTemporalGraphical(nn.Module):
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Output: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """
    def __init__(self,
                 time_dim,
                 joints_dim
    ):
        super(ConvTemporalGraphical,self).__init__()
        
        self.A=nn.Parameter(torch.FloatTensor(time_dim, joints_dim,joints_dim)) #learnable, graph-agnostic 3-d adjacency matrix(or edge importance matrix)
        stdv = 1. / math.sqrt(self.A.size(1))
        self.A.data.uniform_(-stdv,stdv)

        self.T=nn.Parameter(torch.FloatTensor(joints_dim , time_dim, time_dim)) 
        stdv = 1. / math.sqrt(self.T.size(1))
        self.T.data.uniform_(-stdv,stdv)
        '''
        self.prelu = nn.PReLU()
        
        self.Z=nn.Parameter(torch.FloatTensor(joints_dim, joints_dim, time_dim, time_dim)) 
        stdv = 1. / math.sqrt(self.Z.size(2))
        self.Z.data.uniform_(-stdv,stdv)
        '''
    def forward(self, x):
        x = torch.einsum('nctv,vtq->ncqv', (x, self.T))
        ## x=self.prelu(x)
        x = torch.einsum('nctv,tvw->nctw', (x, self.A))
        ## x = torch.einsum('nctv,wvtq->ncqw', (x, self.Z))
        return x.contiguous() 


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=78):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        # self.act_f = nn.Tanh()
        self.act_f = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):

        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=78):
        """

        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN, self).__init__()
        self.num_stage = num_stage
        self.hidden_feature = hidden_feature
        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gc2 = GC_Block(in_features=hidden_feature, p_dropout=p_dropout, node_n=node_n)
        self.gc3 = GC_Block(in_features=hidden_feature, p_dropout=p_dropout, node_n=node_n)
        self.gc4 = GC_Block(in_features=hidden_feature, p_dropout=p_dropout, node_n=node_n)
        self.gc5 = GC_Block(in_features=hidden_feature, p_dropout=p_dropout, node_n=node_n)
        self.gc6 = GC_Block(in_features=hidden_feature, p_dropout=p_dropout, node_n=node_n)
        self.gc7 = GC_Block(in_features=hidden_feature, p_dropout=p_dropout, node_n=node_n)
        self.gc8 = GC_Block(in_features=hidden_feature, p_dropout=p_dropout, node_n=node_n)
        self.gc9 = GC_Block(in_features=hidden_feature, p_dropout=p_dropout, node_n=node_n)
        self.gc10 = GC_Block(in_features=hidden_feature, p_dropout=p_dropout, node_n=node_n)
        self.gc11 = GC_Block(in_features=hidden_feature, p_dropout=p_dropout, node_n=node_n)
        self.gc12 = GC_Block(in_features=hidden_feature, p_dropout=p_dropout, node_n=node_n)
        # self.gc13 = GC_Block(in_features=hidden_feature, p_dropout=p_dropout, node_n=node_n)

        self.gc13 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)
        self.bn13 = nn.BatchNorm1d(node_n * input_feature)

        self.do = nn.Dropout(p_dropout)
        # self.act_f = nn.Tanh()
        self.act_f = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):

        y1 = self.gc1(x)
        b, n, f = y1.shape
        y1 = self.bn1(y1.view(b, -1)).view(b, n, f)
        y1 = self.act_f(y1)
        y1 = self.do(y1)

        y2 = self.gc2(y1)
        y3 = self.gc3(y2)
        y4 = self.gc4(y3)
        y5 = self.gc5(y4)
        y6 = self.gc6(y5)
        y7 = self.gc7(y6)
        y8 = self.gc8(y7 + y6)
        y9 = self.gc9(y8 + y5)
        y10 = self.gc10(y9 + y4)
        y11 = self.gc11(y10 + y3)
        y12 = self.gc12(y11 + y2)

        y13 = self.gc13(y12 + y1)

        # y14 = self.gc14(y13)
       # print("GCN DOWN")

        return y13 + x



