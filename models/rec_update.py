
from models.Updateblock import *
from models.lookup import *
from torchsummary import summary
import torch.nn as nn
import torch
import math
from .submodule import *


class gru_fix(nn.Module):
    def __init__(self,hidden_dim=320,up_factor=4):
        super(gru_fix, self).__init__()
        self.up_factor = up_factor
        self.hidden_dim = hidden_dim
        self.update_block = updateblock(self.hidden_dim, self.up_factor)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, cost,context,initialdisp =0.0,att=None,aggreator=None):

        B,D,H,W = cost.shape
        #B D H W -->
        cost_input = cost.permute(0,2,3,1).reshape(B*H*W,1,1,D)

        initial_hiddel = context

        flow_predictions = []
        coords0, coords1 = self.initialize_flow(context)

        flow = coords1-coords0

        flow[:,0]=initialdisp

        for itr in range(5):
            flow = flow.detach()
            if itr ==0:

                x= look(flow,cost_input)

                hidden , delta_flow ,mask= self.update_block(x, initial_hiddel,flow,context,att,aggreator)


                delta_flow[:,1]=0.0
                flow = flow+delta_flow
                flowup=self.upsample_flow(flow,mask,self.up_factor)
                flow_predictions.append(flowup[:,0])
            else:
                x=look(flow,cost_input)
                hidden,delta_flow ,mask= self.update_block(x,hidden,flow,context,att,aggreator)
                delta_flow[:, 1] = 0.0
                flow = flow+delta_flow
                flowup=self.upsample_flow(flow,mask,self.up_factor)

                flow_predictions.append(flowup[:, 0]) # N , H, W

        return flow_predictions


    def upsample_flow(self, flow, mask,factor=4):
        """[H/factor, W/factor, 2] -> [H, W, 2]"""
        N, D, H, W = flow.shape
        factor = factor
        mask = mask.view(N, 1, 9, factor, factor, H, W)  # 147456
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor * H, factor * W)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            if isinstance(m, nn.BatchNorm3d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0, coords1
if __name__ == "__main__":
    model = gru_fix()

    summary(model,input_size=[(32,256,512),(1,256,512),(32,256,512)],device='cpu',batch_size=1)