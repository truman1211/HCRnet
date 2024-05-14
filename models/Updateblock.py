import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ConvGRU import *
import numpy as np


class FlowHead(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, output_dim=2):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class updateblock(nn.Module):

    def __init__(self,hidden_dim=32,up_factor=4): #32+9+2=
        super(updateblock, self).__init__()
        self.hidden_dim = hidden_dim

        self.mask = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim//2, up_factor**2*9, 1, padding=0))

        self.gru = convGRU(hidden_dim=self.hidden_dim, input_dim=self.hidden_dim+9+2+11)


        self.flow_head = FlowHead(input_dim=self.hidden_dim,hidden_dim=self.hidden_dim//2)



    def forward(self,x,hidden,flow,context_feature,att=None,aggregator=None):

        hidden = self.gru(x,hidden,flow,context_feature,att,aggregator)

        delta_flow = self.flow_head(hidden)

        mask = self.mask(hidden)*0.25


        return  hidden , delta_flow ,mask
