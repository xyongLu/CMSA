import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools
import copy
import numpy as np
from typing import Tuple
from timm.models.layers import DropPath, trunc_normal_, create_pool2d, create_conv2d
from timm.models.registry import register_model

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class Conv2d_BN(nn.Module):
    def __init__(self, in_features, out_features=None, kernel_size=3, stride=1, padding=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_features)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)
    
        # global FLOPS_COUNTER
        # output_points = ((resolution + 2 * padding - dilation *
        #                   (ks - 1) - 1) // stride + 1)**2
        # FLOPS_COUNTER += a * b * output_points * (ks**2) // groups
    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m
    
    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ReparamBlock(nn.Module):
    """ ReparamBlock building block.
        This block has a multi-branched architecture at train-time
        and plain-CNN style architecture at inference time
        Following works like `An Improved One millisecond Mobile Backbone` -
        https://arxiv.org/pdf/2206.04040.pdf
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 hybrid_sale: bool = True,
                 rbr_skip: bool = True,
                 inference_mode: bool = False,
                 num_conv_branches: int = 1,
                 activation= nn.Identity()) -> None:
        """ 
        :param in_channels: Number of channels in the input.
        :param out_channels: Number of channels produced by the block.
        :param kernel_size: Size of the convolution kernel.
        :param stride: Stride size.
        :param padding: Zero-padding size.
        :param dilation: Kernel dilation factor.
        :param groups: Group number.
        :param hybrid_sale: If True, use multi-branches with hybrid-scale receptive field
        :param rbr_skip: If True, add one skip branch 
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
        super(ReparamBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        # Check if SE-ReLU is requested
        self.se = nn.Identity()
        self.activation = activation #nn.ReLU()
        if inference_mode:
            self.reparam_conv = nn.Conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=True)
        else:
            # Re-parameterizable skip connection
            self.rbr_skip = nn.BatchNorm2d(num_features=in_channels) \
                if rbr_skip and out_channels == in_channels and stride == 1 else None


            # Re-parameterizable conv branches
            rbr_conv = list()
            for _ in range(self.num_conv_branches):
                rbr_conv.append(self._conv_bn(kernel_size=kernel_size, padding=padding))
            self.rbr_conv = nn.ModuleList(rbr_conv)

            # Re-parameterizable scale branch
            self.rbr_scale = None
            if kernel_size > 1 and hybrid_sale:
                self.rbr_scale = self._conv_bn(kernel_size=1, padding=0)
         
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))
        
        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches
        out = scale_out + identity_out
        for ix in range(self.num_conv_branches):
            out += self.rbr_conv[ix](x)

        return self.activation(out)
    
    def _conv_bn(self,
                 kernel_size: int,
                 padding: int, 
                  ) -> nn.Sequential:
        """ Helper method to construct conv-batchnorm layers.
        :param kernel_size: Size of the convolution kernel.
        :param padding: Zero-padding size.
        :return: Conv-BN module.
        """
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              kernel_size=kernel_size,
                                              stride=self.stride,
                                              padding=padding,
                                              groups=self.groups,
                                              bias=False))
        mod_list.add_module('bn', nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list

    def reparameterize(self):
        """ Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(in_channels=self.rbr_conv[0].conv.in_channels,
                                      out_channels=self.rbr_conv[0].conv.out_channels,
                                      kernel_size=self.rbr_conv[0].conv.kernel_size,
                                      stride=self.rbr_conv[0].conv.stride,
                                      padding=self.rbr_conv[0].conv.padding,
                                      dilation=self.rbr_conv[0].conv.dilation,
                                      groups=self.rbr_conv[0].conv.groups,
                                      bias=True)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_conv')
        self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        :param branch:
        :return: Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                 self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

class SpatialFusionModules(nn.Module):
    def __init__(self, in_features, out_features=None, act_layer=nn.GELU):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features

        self.dwconv_bn = Conv2d_BN(self.in_features, self.in_features, kernel_size=3,  stride=1, padding=1, groups=self.in_features)
        self.pwconv_bn = Conv2d_BN(self.in_features, self.out_features, kernel_size=1,  stride=1, padding=0)
        
        self.act = act_layer()

    def forward(self, x):
        x = x + self.dwconv_bn(x)
        x = self.act(x)
        x = self.pwconv_bn(x)

        return x 

class Attention(nn.Module):
    """
    LSA: self attention within a group
    """
    def __init__(self, key_dim, num_heads, ws_reso, ws_ratio, attn_ratio, qkv_bias=False, qk_scale=None, attn_drop=0.):
        super(Attention, self).__init__()
        # assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.ws_reso = ws_reso
        self.q_reso_h, self.q_reso_w = ws_reso
        self.kv_reso_h, self.kv_reso_w = ws_reso[0]//ws_ratio, ws_reso[1]//ws_ratio
        # self.kv_reso = ws_reso // ws_ratio
        self.key_dim = key_dim#dim // num_heads,  qk_key_dim
        self.scale = qk_scale or self.key_dim ** -0.5
        self.num_heads = num_heads

        self.nh_kd = key_dim * num_heads
        self.d =  int(attn_ratio * key_dim)
        self.dh = self.d  * num_heads
        # self.qkv_dim = self.dh + self.nh_kd * 2
        # self.qkv_kd = self.d + key_dim*2

        # self.qkv = nn.Conv2d(dim, self.qkv_dim, kernel_size=1,  stride=1, padding=0)
        # self.kv = nn.Conv2d(dim, self.nh_kd+self.dh, kernel_size=1,  stride=1, padding=0)
        # self.q = nn.Conv2d(dim, self.nh_kd, kernel_size=1,  stride=1, padding=0)
        
        self.attn_drop = nn.Dropout(attn_drop)

        points_q = list(itertools.product(range(self.q_reso_h), range(self.q_reso_w)))
        points_kv = list(itertools.product(range(self.kv_reso_h), range(self.kv_reso_w)))
        N_q = len(points_q)
        N_kv = len(points_kv)
        attention_offsets = {}
        idxs = []
        for p1 in points_q:
            for p2 in points_kv:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N_q, N_kv))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, q, kv):
        B, _, H, W = q.shape  # kv: B,_, H/2, W/2
        nH = H // self.q_reso_h
        nW = W // self.q_reso_w

        kv = kv.reshape(B,  self.num_heads, self.key_dim+self.d, nH, self.kv_reso_h, nW, self.kv_reso_w).permute(0, 3, 5, 1, 4, 6, 2).contiguous().reshape(B, nH*nW, self.num_heads, self.kv_reso_h*self.kv_reso_w, -1)
        k, v = kv.split([self.key_dim, self.d], dim=4)
        q = q.reshape(B,  self.num_heads, self.key_dim, nH, self.q_reso_h, nW, self.q_reso_w).permute(0, 3, 5, 1, 4, 6, 2).contiguous().reshape(B, nH*nW, self.num_heads, self.q_reso_h*self.q_reso_w, self.key_dim)

        attn = ( (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws
                +
                (self.attention_biases[:, self.attention_bias_idxs]
                 if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)  # B, hw, n_head, ws_q*ws_q, ws_kv*ws_kv
        attn = self.attn_drop(attn)  # group attn @ v-> B, hw, n_head, ws*ws, head_dim -> (t(2,3)) B, hw, ws*ws, n_head,  head_dim
        out = (attn @ v).permute(0, 2, 4, 1, 3).contiguous().reshape(B, self.dh, nH, nW,  self.q_reso_h,  self.q_reso_w)
        out = out.transpose(3, 4).contiguous().reshape(B, -1, H, W)
        return out

class MultiScaleAttention(torch.nn.Module):
    def __init__(self, dim=384, key_dim=32, br_num_heads=[1,1,1], br_wss_resos=[16,8,4], br_wss_ratios=[2,2,2], attn_ratio=2, proj_drop=0., inference_mode=False):
        super().__init__()
        self.br_wss_resos = br_wss_resos#[16, 8, 4]
        self.br_wss_ratios = br_wss_ratios#[8, 4, 2]
        self.key_dim = key_dim
        self.br_num_heads = br_num_heads
        self.num_wss_resos = len(self.br_wss_resos)
        
        heads = sum(br_num_heads)
        self.d = int(attn_ratio * key_dim)
        self.dh = self.d * heads
        self.nh_kd = key_dim * heads

        self.kv_kd = self.d + key_dim
        self.kv_dim = self.dh + self.nh_kd
        self.q = nn.Conv2d(dim, self.nh_kd, kernel_size=1, stride=1, padding=0)
        self.kv = nn.Conv2d(dim, self.kv_dim, kernel_size=1, stride=1, padding=0)

        self.br_kv_dims = []
        self.br_q_dims = []
        self.attn_modules = nn.ModuleList()
        self.channel_fusion_modules = nn.ModuleList()
        self.spatial_fusion_modules = nn.ModuleList()
        self.pools = nn.ModuleList()

        self.attn_modules.append(Attention(key_dim=key_dim, num_heads=self.br_num_heads[0],  ws_reso=self.br_wss_resos[0], ws_ratio=self.br_wss_ratios[0], attn_ratio=attn_ratio))
        self.pools.append(nn.AvgPool2d(kernel_size=self.br_wss_ratios[0], stride=self.br_wss_ratios[0]))
        self.spatial_fusion_modules.append(SpatialFusionModules(in_features= self.br_num_heads[0]*self.kv_kd, out_features=self.br_num_heads[0]*self.kv_kd))
        self.br_kv_dims.append(self.br_num_heads[0]*self.kv_kd)
        self.br_q_dims.append(self.br_num_heads[0]*self.key_dim)

        # if self.num_wss_resos>1:
        for i in range(1, self.num_wss_resos):
            self.br_kv_dims.append(self.br_num_heads[i]*self.kv_kd)
            self.br_q_dims.append(self.br_num_heads[i]*self.key_dim)
            self.channel_fusion_modules.append(Conv2d_BN(self.br_num_heads[i]*self.kv_kd + self.br_num_heads[i-1]*self.d, self.br_num_heads[i]*self.kv_kd, kernel_size=1, stride=1, padding=0))
            self.attn_modules.append(Attention(key_dim=key_dim, num_heads=self.br_num_heads[i],  ws_reso=self.br_wss_resos[i], ws_ratio=self.br_wss_ratios[i], attn_ratio=attn_ratio))
            self.pools.append(nn.AvgPool2d(kernel_size=self.br_wss_ratios[i], stride=self.br_wss_ratios[i]))
            self.spatial_fusion_modules.append(SpatialFusionModules(in_features= self.br_num_heads[i]*self.kv_kd, out_features=self.br_num_heads[i]*self.kv_kd))

        self.linear = nn.Linear(self.dh, dim)
        self.drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape

        # if self.num_wss_resos !=1:
        kv_s = self.kv(x).split(self.br_kv_dims, dim=1) 
        q_s = self.q(x).split(self.br_q_dims, dim=1)

        #scale1:
        kv_s0 = self.pools[0](self.spatial_fusion_modules[0](kv_s[0]))
        out_s0 = self.attn_modules[0](q_s[0], kv_s0)  
        out_ss = [out_s0]
        
        for i in range(1, self.num_wss_resos):
            kv_si = torch.cat((kv_s[i], out_ss[i-1]), dim=1)
            kv_si = self.channel_fusion_modules[i-1](kv_si)
            kv_si_ = self.pools[i](self.spatial_fusion_modules[i](kv_si))
            out_si = self.attn_modules[i](q_s[i], kv_si_)
            out_ss.append(out_si)
        
        attn_out = torch.cat(tuple(out_ss), dim=1).flatten(2).transpose(1, 2)
        out = self.linear(attn_out).transpose(1, 2).contiguous().reshape(B, C, H, W)
        out = self.drop(out)
        return out

class CMSAFormerBlock(nn.Module):  
    def __init__(self, dim, key_dim, br_num_heads, br_wss_resos, br_wss_ratios, attn_ratio, mlp_ratio=4.,  drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, inference_mode=False):
        super().__init__()
        self.dim = dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.pos_embed = ReparamBlock(in_channels=self.dim,
                                        out_channels=self.dim,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=self.dim,
                                        inference_mode=False,
                                        num_conv_branches=1,
                                        activation=nn.GELU()
                                        )

        self.attn = MultiScaleAttention(dim=self.dim, key_dim=key_dim, br_num_heads=br_num_heads, br_wss_resos=br_wss_resos, br_wss_ratios=br_wss_ratios, attn_ratio=attn_ratio)
        self.norm1 = nn.BatchNorm2d(self.dim)
        self.mlp = Mlp(in_features=self.dim, hidden_features=self.dim*mlp_ratio, out_features=self.dim, act_layer=act_layer, drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.pos_embed(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).contiguous().reshape(B, C, H, W)

        return x


class FineGrainedMappingModule(nn.Module):
    def __init__(self, embed_dims, out_dim, resolution):
        super(FineGrainedMappingModule, self).__init__()            
        self.embed_dims = embed_dims
        self.in_features = sum(embed_dims)
        self.num_stages = len(embed_dims)
        self.out_features = out_dim
        self.size = resolution

        self.csMixer = nn.Sequential(
            Conv2d_BN(in_features=self.in_features, out_features=self.in_features//2, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            Conv2d_BN(in_features=self.in_features//2, out_features=self.out_features, kernel_size=1, stride=1, padding=0),
        )

        self.norm0 = nn.BatchNorm2d(self.embed_dims[0])
        self.convTrans1= nn.ConvTranspose2d(self.embed_dims[1], self.embed_dims[1], kernel_size=2, stride=2, groups=self.embed_dims[1])
        self.norm1 = nn.BatchNorm2d(self.embed_dims[1])
        # self.relu = nn.ReLU(inplace=True)
        self.convTrans2= nn.ConvTranspose2d(self.embed_dims[2], self.embed_dims[2], kernel_size=4, stride=4, groups=self.embed_dims[2])
        self.norm2 = nn.BatchNorm2d(self.embed_dims[2])

    def forward(self,xs):

        x_s0 = xs[0]
        x_s1 = self.norm1(self.convTrans1(xs[1]))
        x_s2 = self.norm2(self.convTrans2(xs[2]))
        xs_ = [x_s0, x_s1, x_s2]

        xs_ = torch.cat(xs_, dim=1)
        out = self.csMixer(xs_)
        return out

class Embedding(nn.Module):
    """ Patch Merging Layer.
    """
    def __init__(self,  in_features, out_features, stride=2):
        super().__init__()
        # self.norm = nn.BatchNorm2d(out_features)
        # self.conv = Conv2d_BN(in_features, out_features, kernel_size=3, stride=2, padding=1)
        
        self.in_features = in_features
        self.out_features = out_features
        self.rep_dwconv = ReparamBlock(in_channels=self.in_features,
                                         out_channels=self.in_features,
                                         kernel_size=3,
                                         stride=stride,
                                         padding=1,
                                         groups=self.in_features,
                                         inference_mode=False,
                                         num_conv_branches=1,
                                         activation=nn.GELU())
        
        self.rep_pwconv = ReparamBlock(in_channels=self.in_features,
                                         out_channels=self.out_features,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         inference_mode=False,
                                         num_conv_branches=1
                                         )

    def forward(self, x):
        # x = self.conv(x)

        x = self.rep_dwconv(x)
        x = self.rep_pwconv(x)

        return x

class Stem(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=32, in_chans=3, embed_dim=64):
        super().__init__()
        self.in_features = in_chans
        self.embed_dim = embed_dim
        self.act = nn.GELU()     
        self.embedding1 = Embedding(self.in_features, self.embed_dim//2, stride=1)
        self.embedding2 = Embedding(self.embed_dim//2, self.embed_dim, stride=1)  

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.embedding1(x)
        x = self.act(x)
        x = self.embedding2(x)
        x = self.act(x)
        return x

class CMSAFormer(nn.Module):
    """CMSAFormer
    """
    def __init__(self, img_size=32, in_chans=3, num_classes=1000, embed_dims=[128, 256, 320], key_dim=32, stage_num_heads=[[2],[4],[8]], stage_wss_resos=[[64,48],[32,24],[16,12]], stage_wss_ratios=[[2],[2],[2]], attn_ratios=[2, 2, 2], mlp_ratio=4, 
                drop_rate=0., attn_drop_rate=0.0, drop_path_rate=0.0, depths_blocks=[2, 3, 2], inference_mode=False):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_stages = len(self.embed_dims)
        self.depths_blocks = depths_blocks
        self.stage_num_heads = stage_num_heads
        self.stage_wss_resos = stage_wss_resos 
        self.stage_wss_ratios = stage_wss_ratios 
        self.attn_ratios = attn_ratios
        
        self.embedding_blocks = nn.ModuleList()
        self.stage_blocks = nn.ModuleList()

        self.stem = Stem(img_size=img_size, in_chans=in_chans, embed_dim=self.embed_dims[0])
        self.embedding_blocks.append(self.stem)
        for i in range(self.num_stages-1):
            self.embedding_blocks.append(Embedding(in_features=self.embed_dims[i], out_features=self.embed_dims[i+1], stride=2))

        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths_blocks))]  # stochastic depth decay rule
        cur = 0
        for i in range(self.num_stages):
            _blocks = nn.Sequential()
            for k in range(self.depths_blocks[i]): 
                _blocks.add_module("trains_block{0}".format(k), CMSAFormerBlock(dim=self.embed_dims[i], key_dim=key_dim, br_num_heads=self.stage_num_heads[i], br_wss_resos=self.stage_wss_resos[i], br_wss_ratios=self.stage_wss_ratios[i],
                        attn_ratio=self.attn_ratios[i], mlp_ratio=mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[cur], inference_mode=inference_mode))
            cur +=  self.depths_blocks[i]
            self.stage_blocks.append(_blocks)

        self.fgmm = FineGrainedMappingModule(embed_dims=self.embed_dims, out_dim=num_classes, resolution=(32,24)) #if input-size<=(64,48), resolution is (32, 24) else (64,48)

        # init weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def forward_features(self, x):
        stages_out = []
        for i in range(self.num_stages):
            x = self.embedding_blocks[i](x)
            x = self.stage_blocks[i](x)
            stages_out.append(x)
        return stages_out

    def forward(self, x):
        xs = self.forward_features(x)

        out = self.fgmm(xs)
        return out


@register_model
def CMSAFormer_S_32(pretrained=False, inference_mode=False, **kwargs): 
    model = CMSAFormer(
        embed_dims=[96, 160, 224], 
        key_dim=16, 
        depths_blocks=[2, 4, 3], 
        stage_num_heads = [[2,2,1], [3, 3], [4, 4]],
        stage_wss_resos = [[(32,24), (16,12), (8,6)],[(16,12), (8,6)], [(8,6), (8,6)]],
        stage_wss_ratios = [[2, 2, 1],[ 2, 2], [1, 2]], 
        attn_ratios=[2, 2, 2], mlp_ratio=4, inference_mode=inference_mode, **kwargs)

    return model

@register_model
def CMSAFormer_B_32(pretrained=False, inference_mode=False, **kwargs): 
    model = CMSAFormer(
        embed_dims=[128, 192, 256], 
        key_dim=16, 
        depths_blocks=[2, 4, 3], 
        stage_num_heads = [[1,2,1], [3, 3], [4, 5]],
        stage_wss_resos = [[(32,24),(16, 12), (8, 6)],[(16,12), (8,6)], [(8,6), (8,6)]],
        stage_wss_ratios = [[2, 2, 1],[2, 2], [1, 2]], 
        attn_ratios=[2, 2, 2], mlp_ratio=4, inference_mode=inference_mode, **kwargs)

    return model

@register_model
def CMSAFormer_L_32(pretrained=False, inference_mode=False, **kwargs): 
    model = CMSAFormer(
        embed_dims=[128, 256, 320], 
        key_dim=16, 
        depths_blocks=[2, 4, 3], 
        stage_num_heads = [[1,2,1], [3, 3], [4, 4]],
        stage_wss_resos = [[ (32,24),(16, 12), (8, 6)],[(16,12), (8,6)], [(8,6), (8,6)]],
        stage_wss_ratios = [[2, 2, 1],[2, 2], [1, 2]], 
        attn_ratios=[2, 2, 2], mlp_ratio=4, inference_mode=inference_mode, **kwargs)

    return model


def reparameterize_model(model: torch.nn.Module) -> nn.Module:
    """ Method returns a model where a multi-branched structure
        used in training is re-parameterized into a single branch
        for inference.

    :param model: MobileOne model in train mode.
    :return: MobileOne model in inference mode.
    """
    # Avoid editing original graph
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'reparameterize'):
            module.reparameterize()
    return model

