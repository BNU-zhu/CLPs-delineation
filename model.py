from torch import nn
import torch
from torch.nn import functional as F
import numpy as np




class Squeeze_Excitation(nn.Module):
    def __init__(self, channel, r=8):
        """
        Based on Squeeze-and-Excitation Networks (SENet) Implementation 
        @ https://github.com/hujie-frank/SENet
        """
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        b, c, _, _ = inputs.shape
        x = self.pool(inputs).view(b, c)
        x = self.net(x).view(b, c, 1, 1)
        x = inputs * x
        return x

class Stem_Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_c),
        )

        self.attn = Squeeze_Excitation(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)
        #y = x + s
        return y

class ResNet_Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        """
        Based on ResNet Implementation 
        @ https://github.com/vipul2001/Modern-CNNs-Implementation/tree/master/Resnet
        """
        super().__init__()

        self.c1 = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_c),
        )

        self.attn = Squeeze_Excitation(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)
        #y = x + s
        return y

######ASPP
"""
Reference the implementation of ASPP in the DeeplabV3+ 
@ https://github.com/VainF/DeepLabV3Plus-Pytorch
"""
class ASPPConv1x1(nn.Sequential):
    def __init__(self, in_channels, out_channels):
                
        modules = [nn.Conv2d(in_channels, out_channels, 1, bias=False),
                   nn.BatchNorm2d(out_channels),
                   nn.ReLU(inplace=True), ]
        super(ASPPConv1x1, self).__init__(*modules)
        pass

    pass
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        
        modules = [nn.Conv2d(in_channels, out_channels, kernel_size=3,
                             padding=dilation, dilation=dilation, bias=False), 
                   nn.BatchNorm2d(out_channels),  
                   nn.ReLU(inplace=True), ]  
        super(ASPPConv, self).__init__(*modules)
        pass

    pass
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        
        modules = [nn.AdaptiveAvgPool2d(1),  
                   nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),  
                   nn.BatchNorm2d(out_channels),  
                   nn.ReLU(inplace=True), ]  
        super(ASPPPooling, self).__init__(*modules)
        pass

    def forward(self, x):
        size = x.shape[-2:]  
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True) 

    pass
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Reference the implementation of ASPP in the DeeplabV3+ 
        @ https://github.com/VainF/DeepLabV3Plus-Pytorch
        """
        super(ASPP, self).__init__()
        modules = [ASPPConv1x1(in_channels, out_channels), 
                   ASPPConv(in_channels, out_channels, dilation=1),  
                   ASPPConv(in_channels, out_channels, dilation=2),  
                   ASPPConv(in_channels, out_channels, dilation=4),  
                   ASPPPooling(in_channels, out_channels), ] 
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout2d(0.5)) 
        pass

    def forward(self, x):
        output = []
        for mod in self.convs:
            output.append(mod(x))
            pass
        x = torch.cat(output, dim=1)
        x = self.project(x)
        return x

    pass

class Attention_Gate(nn.Module):
    def __init__(self, in_c):
        super(Attention_Gate, self).__init__()
        out_c = in_c[1]
        self.W_g = nn.Sequential(
            nn.Conv2d(in_c[0], out_c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_c)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(in_c[1], out_c, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_c)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(out_c, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):# g: feature map of the previous-stage Decodeblock
                            # x: feature map of the skip connection
        g1 = self.W_g(g)
        x1 = self.W_x(x) 
        psi = self.relu(g1 + x1)
        # channel to 1 and Sigmoid to get the weight map,
        psi = self.psi(psi)
        # adjust the feature map of the skip connection
        return x * psi

class Decoder_Attention(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.a1 = Attention_Gate(in_c)
        self.r1 = ResNet_Block(in_c[0]+in_c[1], out_c, stride=1)

    def forward(self, g, x):# g: feature map of the previous-stage Decodeblock
                            # x: feature map of the skip connection
        g = self.up(g) # Upsample the feature map g
        d = self.a1(g, x) # weighted adjust the feature map x
        d = torch.cat([d, g], axis=1) # concatenate the two feature maps
        d = self.r1(d) # decoded by ResNet_Block
        return d


class CDCM(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(CDCM, self).__init__()

        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=5, padding=5, bias=False)
        self.conv2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=7, padding=7, bias=False)
        self.conv2_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=9, padding=9, bias=False)
        self.conv2_4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=11, padding=11, bias=False)
        nn.init.constant_(self.conv1.bias, 0)
        
    def forward(self, x):
        x = self.relu1(x)
        x = self.conv1(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x4 = self.conv2_4(x)
        return x1 + x2 + x3 + x4

class HAFM(nn.Module):
    def __init__(self, input_channels):
        super(HAFM, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        # Calculating Attention Maps
        transformed_feature = self.conv2(self.activ(self.conv1(x)))

        # The normalization of the attention map
        #attention_weights = F.sigmoid(transformed_feature)#, dim=1
        attention_weights = F.softmax(transformed_feature,dim=1)

        # Calculating the fused boundary result
        fused_feature = torch.sum(attention_weights * x, dim=1, keepdim=True) 
        
        return fused_feature





######################################


class Our_Model(nn.Module):
    def __init__(self,num_classes=2, add_output=True, bilinear=False, num_filters=32,is_deconv=False):
        super(Our_Model, self).__init__()
        # lr 1 2 decay 1 0
        self.cov1 = Stem_Block(3, 64, stride=1)
        self.cov2 = ResNet_Block(64, 128, stride=2)
        self.cov3 = ResNet_Block(128, 256, stride=2)
        self.cov4 = ResNet_Block(256, 512, stride=2)
        self.cov5 = ResNet_Block(512, 1024, stride=2)
        
        self.aspp1 = ASPP(512, 1024)

        self.dec_att1 = Decoder_Attention([1024, 256], 512)
        self.dec_att2 = Decoder_Attention([512, 128], 256)
        self.dec_att3 = Decoder_Attention([256, 64], 128)

        self.aspp2 = ASPP(128, 64)
        self.maskout = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.disout = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        
        
        self.bou_CDCM1 = CDCM(64, 21)
        self.bou_CDCM2 = CDCM(128, 21)
        self.bou_CDCM3 = CDCM(256, 21)
        self.bou_CDCM4 = CDCM(512, 21)
        self.bou_CDCM5 = CDCM(1024, 21)
        
        self.boufea1 = nn.Conv2d(21, 1, 1)
        self.boufea2 = nn.Conv2d(21, 1, 1)
        self.boufea3 = nn.Conv2d(21, 1, 1)
        self.boufea4 = nn.Conv2d(21, 1, 1)
        self.boufea5 = nn.Conv2d(21, 1, 1)
        
        self.attentionfusion = HAFM(5)
        
    def forward(self, x):
        
        H, W = x.shape[2], x.shape[3]
        
        cov1 = self.cov1(x)
        cov2 = self.cov2(cov1)
        cov3 = self.cov3(cov2)
        cov4 = self.cov4(cov3)
        cov5 = self.cov5(cov4)
        bride1 = self.aspp1(cov4)

        mask_dec1 = self.dec_att1(bride1, cov3)
        mask_dec2 = self.dec_att2(mask_dec1, cov2)
        mask_dec3 = self.dec_att3(mask_dec2, cov1)

        mask_aspp2 = self.aspp2(mask_dec3)
        mask_output = self.maskout(mask_aspp2)
        
        dis_output = self.disout(mask_aspp2)
        dis_output = torch.sigmoid(dis_output)
        
        bou_CDCM1 = self.bou_CDCM1(cov1)
        bou_CDCM2 = self.bou_CDCM2(cov2)
        bou_CDCM3 = self.bou_CDCM3(cov3)
        bou_CDCM4 = self.bou_CDCM4(cov4)
        bou_CDCM5 = self.bou_CDCM5(cov5)
        
        boufea1 = self.boufea1(bou_CDCM1)
        boufea2 = self.boufea2(bou_CDCM2)
        boufea3 = self.boufea3(bou_CDCM3)
        boufea4 = self.boufea4(bou_CDCM4)
        boufea5 = self.boufea5(bou_CDCM5)   
       
        bou1 = F.interpolate(boufea1, size=(H, W), mode='bilinear', align_corners=True)
        bou2 = F.interpolate(boufea2, size=(H, W), mode='bilinear', align_corners=True)
        bou3 = F.interpolate(boufea3, size=(H, W), mode='bilinear', align_corners=True)
        bou4 = F.interpolate(boufea4, size=(H, W), mode='bilinear', align_corners=True)
        bou5 = F.interpolate(boufea5, size=(H, W), mode='bilinear', align_corners=True)

        fusecat = torch.cat((bou1, bou2, bou3, bou4, bou5), dim=1)
        bou_final = self.attentionfusion(fusecat)
        bou_results = [bou1, bou2, bou3, bou4, bou5, bou_final]
        bou_results = [torch.sigmoid(r) for r in bou_results]
        
        return [mask_output,bou_results,dis_output]