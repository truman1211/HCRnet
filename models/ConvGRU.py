import torch.nn as nn
import torch
import torch.nn.functional as F
class BasicMotionEncoder(nn.Module):
    def __init__(self):
        super(BasicMotionEncoder, self).__init__()

        self.convc1 = nn.Conv2d(9, 16, 1, padding=0)
        self.convc2 = nn.Conv2d(16, 16, 3, padding=1)

        self.convf1 = nn.Conv2d(2, 16, 7, padding=3)
        self.convf2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv = nn.Conv2d(32, 9, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class convGRU(nn.Module):
    def __init__(self,hidden_dim, input_dim):
        super(convGRU, self).__init__()

        self.convz = nn.Conv2d(hidden_dim+input_dim , hidden_dim , kernel_size=3,padding=1,bias=False)
        self.bn1 = nn.InstanceNorm2d(hidden_dim)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim,kernel_size=3,padding=1,bias=False)
        self.bn2=nn.InstanceNorm2d(hidden_dim)

        self.convh_hat = nn.Conv2d(hidden_dim+input_dim - 11,hidden_dim,kernel_size=3,padding=1,bias=False)

        self.bn3=nn.InstanceNorm2d(hidden_dim)

        self.motionencoder = BasicMotionEncoder()

    def forward(self,x,h,flow,context_feature,att,aggregator):

        motionencode = self.motionencoder(flow,x) #9+2

        cost_global = aggregator(att,motionencode)

        hx = torch.cat([h,motionencode,cost_global,context_feature] , dim=1) #128 + 128 +11 +11

        z= torch.sigmoid(self.bn1(self.convz(hx)))

        r=torch.sigmoid(self.bn2(self.convr(hx)))

        q= torch.tanh(self.bn3(self.convh_hat( torch.cat([r*h,x,flow,context_feature] ,dim=1)  )))

        h = (1-z)*h+z*q

        return h