from __future__ import print_function

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.submodule import *
from models.rec_update import gru_fix
from models.psmnet import PSMNet
from models.gma import *
import math


class feature_extraction(nn.Module):
    def __init__(self, concat_feature=False, concat_feature_channel=12):
        super(feature_extraction, self).__init__()
        self.concat_feature = concat_feature

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 128, 8, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        if self.concat_feature:
            self.lastconv = nn.Sequential(convbn(128*3, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        gwc_feature = torch.cat((l2, l3, l4), dim=1)# c=320

        if not self.concat_feature:
            return {"gwc_feature": gwc_feature}
        else:
            concat_feature = self.lastconv(gwc_feature)#32
            return {"gwc_feature": gwc_feature, "concat_feature": concat_feature}


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv4,conv5,conv6

def show_feature_map(feature_map,name):
    feature_map = feature_map.squeeze(0)
    feature_map = feature_map.cpu().numpy()
    feature_map_num = int(feature_map.shape[0])
    row_num = int(np.ceil(np.sqrt(feature_map_num)))
    plt.figure()
    for index in range(1,feature_map_num+1):
        plt.subplot(row_num,row_num,index)
        plt.imshow(feature_map[index-1],cmap='magma_r')
        plt.axis('off')
    #plt.show()
    plt.savefig(name)

class GwcNet(nn.Module):
    def __init__(self, maxdisp, use_concat_volume=False):
        super(GwcNet, self).__init__()
        self.maxdisp = maxdisp
        self.use_concat_volume = use_concat_volume

        self.num_groups = 32


        if self.use_concat_volume:
            self.concat_channels = 32
            self.feature_extraction = feature_extraction(concat_feature=True,
                                                         concat_feature_channel=self.concat_channels)
        else:
            self.concat_channels = 0
            self.feature_extraction = feature_extraction(concat_feature=False)

        self.dres0 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels * 2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)
        self.dres3 = hourglass(32)
        self.dres4 = hourglass(32)


        self.classif1 = nn.Sequential(convbn_3d(128, 64, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(64, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.gru1 = gru_fix(hidden_dim=128,up_factor=4)

        self.att1 = Attention( dim=128, heads=4, max_pos_size=160, dim_head=128)
        self.aggregator1 = Aggregate(dim=11,dim_head=128,heads=4)



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, left, right):
        self.gru1.freeze_bn()


        features_left = self.feature_extraction(left)
        features_right = self.feature_extraction(right)

        context1 = features_left["gwc_feature"][:, -128:]  # 1/4
        context1 = F.avg_pool2d(context1, 4, stride=4)  # 1/6 # N C H W

        context2 = features_left["gwc_feature"][:, 128:256]
        context2 = F.avg_pool2d(context2, 2, stride=2)
        context3 = features_left["gwc_feature"][:, :128]


        att1 = self.att1(torch.tanh(context1))
        att2 = self.att1(torch.tanh(context2))
        att3 = self.att1(torch.tanh(context3))


        gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4,
                                      self.num_groups)
        if self.use_concat_volume:
            concat_volume = build_concat_volume(features_left["concat_feature"], features_right["concat_feature"],
                                                self.maxdisp // 4)

            volume = torch.cat((gwc_volume, concat_volume), 1)
        else:
            volume = gwc_volume

        cost0 = self.dres0(volume)

        cost0 = self.dres1(cost0) + cost0

        out1_16,out2,out3 = self.dres2(cost0)
        out1,out2_8,out3 = self.dres3(out3)
        out1,out2,out3_4=self.dres4(out3)

        cost1 = self.classif1(out1_16)
        cost2 = self.classif2(out2_8)
        cost3 = self.classif3(out3_4)

        cost1 = torch.squeeze(cost1, 1)
        cost2 = torch.squeeze(cost2, 1)
        cost3 = torch.squeeze(cost3,1)

        pred1_list=[]


        pred1 = self.gru1(cost1, context1,initialdisp=0.0,att=att1, aggreator=self.aggregator1)  # 1/4
        for pred in pred1:
            pred = pred.unsqueeze(1)
            pred = F.upsample(pred*4, scale_factor=4, mode='bilinear', align_corners=False)
            pred = pred.squeeze(1)
            pred1_list.append(pred)

        initial_disp = pred1[-1]
        initial_disp = initial_disp.unsqueeze(1)  # 1/4
        initial_disp = F.interpolate(initial_disp*0.5, scale_factor=0.5, mode='bilinear', align_corners=False)   # 1/8
        initial_disp = initial_disp.squeeze(1)
        pred2_list=[]

        pred2 = self.gru1(cost2, context2,initial_disp,att2,self.aggregator1) #1/2

        for pred in pred2:
            pred = pred.unsqueeze(1)
            pred = F.upsample(pred*2, scale_factor=2, mode='bilinear', align_corners=False)
            pred = pred.squeeze(1)
            pred2_list.append(pred)

        initial_disp = pred2[-1]
        initial_disp = initial_disp.unsqueeze(1)  # 1/2
        initial_disp = F.interpolate(initial_disp * 0.5, scale_factor=0.5, mode='bilinear', align_corners=False)  # 1/4
        initial_disp = initial_disp.squeeze(1)

        pred3_list = self.gru1(cost3,context3,initial_disp,att3,self.aggregator1)


        if self.training:


            return pred1_list, pred2_list,pred3_list

        else:

            return pred3_list




def GwcNet_G(d):
    return GwcNet(d, use_concat_volume=False)


def GwcNet_GC(d):
    return GwcNet(d, use_concat_volume=False)

def PSM_HCR(d):
    return PSMNet(d)


