import torch
from timm.models.resnet import resnet50d, resnet101d
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import BasicBlock
from torchvision.models.segmentation.deeplabv3 import ASPP
import sys
import functools
sys.path.append('/home/shenjj/GatedSCNN/')
from base_model import BaseModel 
from blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_vit,
)

##########################
#_make_fusion_block_1
def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, spacial_dim_: int, embed_dim: int, num_heads: int, output_dim: int = None,size: int= None,):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim * spacial_dim_ + size*size, embed_dim) / embed_dim ** 0.5)
        #self.positional_embedding = nn.Parameter(torch.randn(spacial_dim * spacial_dim_, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim).cuda()
        self.q_proj = nn.Linear(embed_dim, embed_dim).cuda()
        self.v_proj = nn.Linear(embed_dim, embed_dim).cuda()
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim).cuda()
        self.num_heads = num_heads
        self.size = size

    def forward(self, x):
        x_ = x
        print(x.shape)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        temp = x.mean(dim=0, keepdim=True)
        if self.size ==1:
           x = torch.cat([temp, x], dim=0)  # (HW+1)NC
        if self.size ==2:
           x = torch.cat([temp,temp,temp,temp, x], dim=0)  # (HW+1)NC
        if self.size ==3:
           x = torch.cat([temp,temp,temp,temp,temp,temp,temp,temp,temp, x], dim=0)  # (HW+1)NC
        if self.size ==6:
           x = torch.cat([temp,temp,temp,temp,temp,temp,temp,temp,temp,temp,temp,temp,temp,temp,temp,temp,temp,temp,temp,temp,temp,temp,temp,temp,temp,temp,temp,temp,temp,temp,temp,temp,temp,temp,temp,temp, x], dim=0)  # (HW+1)NC   
        x = x + self.positional_embedding[:, None, :].to(x.dtype).cuda()  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        print('att',x.shape)
        print('x[0]',x[self.size*self.size].shape)
        #t = x.reshape(x.shape[0],x.shape[1]*x.shape[2])
        #t = torch.mean(x,0)
        #t =t.reshape(x.shape[1], x.shape[2], 1, 1) 
        if self.size ==1:
           t = x[0].reshape(x.shape[1], x.shape[2], self.size,self.size)  # (HW+1)NC
        if self.size ==2: 
           t_ = torch.cat([x[0],x[1],x[2],x[3]], dim=0)  # (HW+1)NC
           t = t_.reshape(x.shape[1], x.shape[2], self.size,self.size)
        if self.size ==3:
           t_ = torch.cat([x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8]], dim=0)  # (HW+1)NC
           t = t_.reshape(x.shape[1], x.shape[2], self.size,self.size)
        if self.size ==6:
           t_ = torch.cat([x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11],x[12],x[13],x[14],x[15],x[16],x[17],x[18],x[19],x[20],x[21],x[22],
           x[23],x[24],x[25],x[26],x[27],x[28],x[29],x[30],x[31],x[32],x[33],x[34],x[35]], dim=0)  # (HW+1)NC
           t = t_.reshape(x.shape[1], x.shape[2], self.size,self.size)
        print(self.size)
        #print('x----',t.shape)
        return t
         

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
    
    
class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        
        print(x.shape)
        return avgpool(x)
    def attpool(self, x, size):
        print(x.shape[2])
        attpool = AttentionPool2d(x.shape[2],x.shape[3],512,1,512,size)
        
        print(x.shape)
        return attpool(x)
                     

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        
        temp1_att = self.attpool(x, 1)
        temp2_att = self.attpool(x, 2)
        temp3_att = self.attpool(x, 3)
        temp4_att = self.attpool(x, 6)
        '''
        temp1 = self.pool(x, 1)
        temp2 = self.pool(x, 2)
        temp3 = self.pool(x, 3)
        temp4 = self.pool(x, 6)
        
        #temp4 = self.attpool(x,6)
        print('temp1',temp1.shape)
        print('temp2',temp2.shape)
        print('temp3',temp3.shape)
        print('temp4',temp4.shape)
        '''
        feat1 = self.upsample(self.conv1(temp1_att), size)
        feat2 = self.upsample(self.conv2(temp2_att), size)
        feat3 = self.upsample(self.conv3(temp3_att), size)
        feat4 = self.upsample(self.conv4(temp4_att), size)
        print('feat4',feat4.shape)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x


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

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        self.ppm = PyramidPooling(features, features)
        
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
        #ppm_out = self.ppm(path_3)
        #path_3 = ppm_out + path_3
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
         
        '''
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        res0 = self.backbone.maxpool(x)

        res1 = self.backbone.layer1(res0)
        res2 = self.backbone.layer2(res1)
        res3 = self.backbone.layer3(res2)
        res4 = self.backbone.layer4(res3)

        return x, res1, res2, res3, res4
        

        out = self.scratch.output_conv(path_1)
        
        return out
        '''
        #out = self.scratch.output_conv(path_1)
        print('layer_1',layer_1.shape)
        print('layer_2',layer_2.shape)
        print('layer_3',layer_3.shape)
        print('layer_4',layer_4.shape)
        #res0 = self.backbone.maxpool(path_1)
        print('x',x.shape)
        print('res0',res0.shape)
        ####################################################xiugaiqian
        res1 = self.backbone.layer1(self.path_1pooling(self.path_1Conv2d(path_1)))
        res2 = self.backbone.layer2(res1)
        res3 = self.backbone.layer3(res2)
        res4 = self.backbone.layer4(res3)
        '''
        res1 = self.path_1pooling(self.path_1Conv2d(path_1))
        res2 = self.path_1pooling(self.path_1Conv2d(path_2))
        res3 = self.path_1pooling(self.path_1Conv2d(path_3))
        res4 = self.path_1pooling(self.path_1Conv2d(path_4))
        #print('out',out.shape)
        print('x',x.shape)
        '''
        print('path_4',path_4.shape)
        print('path_3',path_3.shape)
        
        print('path_2',path_2.shape)
        print('path_1',path_1.shape)
        print('layer_4_rn',layer_4_rn.shape)
        print('layer_3_rn',layer_3_rn.shape)
        print('layer_2_rn',layer_2_rn.shape)
        print('layer_1_rn',layer_1_rn.shape)
        print('res1',res1.shape)
        print('res1',res2.shape)
        print('res1',res3.shape)
        print('res1',res4.shape)
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
#Transformer block
class RegularStream(DPT):
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
        
##########################

class RegularStream0(nn.Module):
    def __init__(self, backbone_type='resnet50'):
        super().__init__()
        if backbone_type == 'resnet50':
            self.backbone = resnet50d(output_stride=8)
        else:
            self.backbone = resnet101d(output_stride=8)

    def forward(self, x):
        print('x',x.shape)
        x = self.backbone.conv1(x)
        print('x',x.shape)
        
        x = self.backbone.bn1(x)
        print('x',x.shape)
        x = self.backbone.act1(x)
        print('x',x.shape)
        res0 = self.backbone.maxpool(x)
        print('x',x.shape)
        print('res0',res0.shape)
        res1 = self.backbone.layer1(res0)
        res2 = self.backbone.layer2(res1)
        res3 = self.backbone.layer3(res2)
        res4 = self.backbone.layer4(res3)
        print('x',x.shape)
        print('res1',res1.shape)
        
        print('res2',res2.shape)
        print('res3',res3.shape)
        print('res4',res4.shape)
        return x, res1, res2, res3, res4


#Boundary-Guided-block 
class ShapeStream(nn.Module):
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
#GatedConv
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

##_make_fusion_block_2
class FeatureFusion(ASPP):
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
############
def apply_kernel(x, kernel, k, normalize=True):
    """apply kernel"""
    # x: (n, c=3, h+2r, w+2r), kernel: (n, k*k, h, w)
    _, _, h, w = kernel.shape
    n, c, _, _ = x.shape
    # kernel = F.sigmoid(kernel) if normalize else kernel
    kernel = F.softmax(kernel, dim=1) if normalize else kernel
    x = F.pad(x, (k // 2,) * 4, 'reflect')  # pad input
    x = F.unfold(x, kernel_size=k).view(n, c, k ** 2, h, w)  # (n,c,k*k,h,w)
    res = (x * kernel.unsqueeze(1)).sum(2, keepdim=False)  # (n,c,k*k,h,w)x(n,1,k*k,h,w)
    return res #£¨n, c, h, w

def get_norm_layer(norm_type):
    """get norm layer"""
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == "none":
        norm_layer = DummyModule
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer

class BasicBlock_(nn.Module):
    """conv+prelu+norm"""

    def __init__(self, dim, norm_type):
        super().__init__()
        norm_layer = get_norm_layer(norm_type)

        blocks = [nn.Conv2d(dim, dim, kernel_size=3, padding=1), norm_layer(dim)]
        blocks += [nn.PReLU(), norm_layer(dim)]
        self.ft = nn.Sequential(*blocks)
        

    def forward(self, x):
        return self.ft(x) + x

class Malleblock(nn.Module):
    def __init__(self, hidden_nc=48, k=3, norm_type='batch', n_blocks=3):
        super(Malleblock, self).__init__()
        norm_layer = get_norm_layer(norm_type)
        self.channle = hidden_nc
        self.k = k

        self.pre1 = nn.Sequential(*[BasicBlock_(hidden_nc, norm_type) for _ in range(n_blocks)])
        self.maxl = nn.MaxPool2d(2, stride=2)
        self.pre2 = nn.Sequential(*[BasicBlock_(hidden_nc, norm_type) for _ in range(n_blocks)])
        self.prek = nn.Sequential(*[nn.Conv2d(hidden_nc, hidden_nc, kernel_size=3, padding=1),
                                        norm_layer(hidden_nc),
                                        nn.PReLU(),
                                        nn.Conv2d(hidden_nc, k**2, kernel_size=3, padding=1)])
        #self.prek.apply(init_weights)
        self.bili = nn.Upsample(mode='bilinear', scale_factor=8, align_corners=True)
        

    def forward(self, x, x_):

        x1 = F.interpolate(x_, scale_factor=0.25)    #(n, c, H/4, W/4)
        #import pdb;pdb.set_trace()
        x1 = self.maxl(self.pre1(x1))   #(n, c, H/8, W/8)
        x1 = self.pre2(x1)  #(n, c, H/8, W/8)
        
        skernel = self.prek(x1)  #(n, k^2, H/8, W/8)
        kernel = self.bili(skernel)  #(n, k^2, H, W)

        return apply_kernel(x, kernel, self.k)
class Malleblock1(nn.Module):
    def __init__(self, hidden_nc=256, k=3, norm_type='batch', n_blocks=3):
        super(Malleblock1, self).__init__()
        norm_layer = get_norm_layer(norm_type)
        self.channle = hidden_nc
        self.k = k

        self.pre1 = nn.Sequential(*[BasicBlock_(hidden_nc, norm_type) for _ in range(n_blocks)])
        self.maxl = nn.MaxPool2d(2, stride=2)
        self.pre2 = nn.Sequential(*[BasicBlock_(hidden_nc, norm_type) for _ in range(n_blocks)])
        self.prek = nn.Sequential(*[nn.Conv2d(hidden_nc, hidden_nc, kernel_size=3, padding=1),
                                        norm_layer(hidden_nc),
                                        nn.PReLU(),
                                        nn.Conv2d(hidden_nc, k**2, kernel_size=3, padding=1)])
        #self.prek.apply(init_weights)
        self.bili = nn.Upsample(mode='bilinear', scale_factor=8, align_corners=True)
        

    def forward(self, x, x_):

        x1 = F.interpolate(x_, scale_factor=0.25)    #(n, c, H/4, W/4)
        #import pdb;pdb.set_trace()
        x1 = self.maxl(self.pre1(x1))   #(n, c, H/8, W/8)
        x1 = self.pre2(x1)  #(n, c, H/8, W/8)
        
        skernel = self.prek(x1)  #(n, k^2, H/8, W/8)
        kernel = self.bili(skernel)  #(n, k^2, H, W)

        return apply_kernel(x, kernel, self.k)

#Boundary-Guided Transformer 
class GatedSCNN(nn.Module):
    def __init__(self, backbone_type='resnet50', num_classes=19):
        super().__init__()

        self.regular_stream = RegularStream(backbone_type)
        self.shape_stream = ShapeStream()
        self.feature_fusion = FeatureFusion(2048, (12, 24, 36), 256)
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

