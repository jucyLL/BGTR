import torch
from timm.models.resnet import resnet50d, resnet101d
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import BasicBlock
from torchvision.models.segmentation.deeplabv3 import ASPP
import sys
import functools
from base_model import BaseModel 
from blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_vit,
)

class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(True)
        )

    def forward(self, x):
        print('x',x.shape)
        return self.conv(x)
    
def _make_fusion_block1(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )    

class DPT(BaseModel):
    def __init__(
        self,
        head,
        features=512,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
    ):

        super(DPT, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11]
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            True,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks["vitb_rn50_384"],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        self.scratch.refinenet1 = _make_fusion_block1(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block1(features, use_bn)
        
        self.scratch.refinenet3 = _make_fusion_block1(features, use_bn)
        
        self.scratch.refinenet4 = _make_fusion_block1(features, use_bn)
        
        
        self.scratch.output_conv = head
        self.backbone = resnet50d(output_stride=8)
        
        self.path_1Conv2d = nn.Conv2d(512, 64, kernel_size=3, padding=1, bias=False)
        self.path_1pooling = nn.AdaptiveAvgPool2d(128)

    def forward(self, x):
        print('x',x.shape)
        x0 = x
        x = self.backbone.conv1(x)
        print('x',x.shape)
        
        x = self.backbone.bn1(x)
        print('x',x.shape)
        x = self.backbone.act1(x)
        print('x',x.shape)
        res0 = self.backbone.maxpool(x)
        
        #x1 =  self.conv1(x)
        #x2 =  self.bn1(x1)
        #x3 =  self.relu1(x2)
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)
        print('x',x.shape)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x0)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        
        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        res1 = self.backbone.layer1(self.path_1pooling(self.path_1Conv2d(path_1)))
        res2 = self.backbone.layer2(res1)
        res3 = self.backbone.layer3(res2)
        res4 = self.backbone.layer4(res3)

        return x, res1, res2, res3, res4


class DPTSegmentationModel(DPT):
    def __init__(self, num_classes):

        features = kwargs["features"] if "features" in kwargs else 256

        kwargs["use_bn"] = True

        head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes, kernel_size=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head, **kwargs)

        self.auxlayer = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, num_classes, kernel_size=1),
        )

class Transformer_B(DPT):
    def __init__(self,non_negative=True):
        features = 512

        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        super().__init__(head)
    def forward(self, x):
        return super().forward(x)
        

class Boundary_G(nn.Module):
    def __init__(self):
        super().__init__()
        self.res2_conv = nn.Conv2d(512, 1, 1)
        self.res3_conv = nn.Conv2d(1024, 1, 1)
        self.res4_conv = nn.Conv2d(2048, 1, 1)
        self.res1 = BasicBlock(64, 64, 1)
        self.res2 = BasicBlock(32, 32, 1)
        self.res3 = BasicBlock(16, 16, 1)
        self.res1_pre = nn.Conv2d(64, 32, 1)
        self.res2_pre = nn.Conv2d(32, 16, 1)
        self.res3_pre = nn.Conv2d(16, 8, 1)
        self.gate1 = GatedConv(32, 32)
        self.gate2 = GatedConv(16, 16)
        self.gate3 = GatedConv(8, 8)
        self.gate = nn.Conv2d(8, 1, 1, bias=False)
        self.fuse = nn.Conv2d(2, 1, 1, bias=False)

    def forward(self, x, res2, res3, res4, grad):
        size = grad.size()[-2:]
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        res2 = F.interpolate(self.res2_conv(res2), size, mode='bilinear', align_corners=True)
        res3 = F.interpolate(self.res3_conv(res3), size, mode='bilinear', align_corners=True)
        res4 = F.interpolate(self.res4_conv(res4), size, mode='bilinear', align_corners=True)

        gate1 = self.gate1(self.res1_pre(self.res1(x)), res2)
        gate2 = self.gate2(self.res2_pre(self.res2(gate1)), res3)
        gate3 = self.gate3(self.res3_pre(self.res3(gate2)), res4)
        gate = torch.sigmoid(self.gate(gate3))
        print('gate',gate.shape)
        print('grad',grad.shape)
        feat = torch.sigmoid(self.fuse(torch.cat((gate, grad), dim=1)))
        return gate, feat

class GatedConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, 1, bias=False)
        self.attention = nn.Sequential(
            nn.BatchNorm2d(in_channels + 1),
            nn.Conv2d(in_channels + 1, in_channels + 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels + 1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, feat, gate):
        attention = self.attention(torch.cat((feat, gate), dim=1))
        out = F.conv2d(feat * (attention + 1), self.weight)
        return out


class _make_fusion_block2(ASPP):
    def __init__(self, in_channels, atrous_rates=(6, 12, 18), out_channels=256):
        # atrous_rates (6, 12, 18) is for stride 16
        super().__init__(in_channels, atrous_rates, out_channels)
        self.shape_conv = nn.Sequential(
            nn.Conv2d(1, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.project = nn.Conv2d((len(atrous_rates) + 3) * out_channels, out_channels, 1, bias=False)
        self.fine = nn.Conv2d(256, 48, kernel_size=1, bias=False)

    def forward(self, res1, res4, feat):
        res = []
        for conv in self.convs:
            res.append(conv(res4))
        res = torch.cat(res, dim=1)
        feat = F.interpolate(feat, res.size()[-2:], mode='bilinear', align_corners=True)
        res = torch.cat((res, self.shape_conv(feat)), dim=1)
        coarse = F.interpolate(self.project(res), res1.size()[-2:], mode='bilinear', align_corners=True)
        fine = self.fine(res1)
        out = torch.cat((coarse, fine), dim=1)
        return out


class BGTR(nn.Module):
    def __init__(self, backbone_type='resnet50', num_classes=19):
        super().__init__()

        self.regular_stream = Transformer_B(backbone_type)
        self.shape_stream = Boundary_G()
        self.feature_fusion = _make_fusion_block2(2048, (12, 24, 36), 256)
        self.seg = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            #nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

    def forward(self, x, grad):
        x, res1, res2, res3, res4 = self.regular_stream(x)
        gate, feat = self.shape_stream(x, res2, res3, res4, grad)
        print('res1',res1.shape)
        print('res4',res4.shape)
        print('feat',feat.shape)
        out = self.feature_fusion(res1, res4, feat)
        seg = F.interpolate(self.seg(out), grad.size()[-2:], mode='bilinear', align_corners=False)
        # [B, N, H, W], [B, 1, H, W]
        return seg, gate


