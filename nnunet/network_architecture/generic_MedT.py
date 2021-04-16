from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv3d, LayerNorm
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import numpy as np
import copy
import ml_collections
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.network_architecture.neural_network import SegmentationNetwork

class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""

def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=(8,16,32),
                 stride=1, bias=False, width=0):
        # in_planes = width =64
        # out_planes = width = 64
        # groups = 8
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes  # planes
        self.out_planes = out_planes  # planes
        self.groups = groups  # 8
        self.group_planes = out_planes // groups  # 8
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size[2-self.width] * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size[2-self.width]).unsqueeze(0)
        key_index = torch.arange(kernel_size[2-self.width]).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size[2-self.width] - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        # pdb.set_trace()
        # N C L H W
        print("x.shape:",x.shape)
        if self.width == 0:  # length
            x = x.permute(0, 2, 3, 1, 4)  # N, L, H ,C, W
        elif self.width == 1:  # height
            x = x.permute(0, 4, 2, 1, 3)  # N, W, L, C, H
        else:  # width
            x = x.permute(0, 3, 4, 1, 2)  # N, H, W, C, L


        N, W, L, C, H = x.shape
        x = x.contiguous().view(N * W * L, C, H)
        print("x.shape:",x.shape)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        print("qkv.shape:",qkv.shape)

        q, k, v = torch.split(qkv.reshape(N * W * L, self.groups, self.group_planes * 2, H),
                              [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)
        print("q.shape:",q.shape)
        print("k.shape:",k.shape)
        print("v.shape:",v.shape)


        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2,
                                                                                       self.kernel_size[2-self.width],
                                                                                       self.kernel_size[2-self.width])

        print("all_beddings.shape:",all_embeddings.shape)


        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                            [self.group_planes // 2, self.group_planes // 2,
                                                             self.group_planes], dim=0)
        print("q_bedding.shape:", q_embedding.shape)
        print("k_bedding.shape:", k_embedding.shape)
        print("v_bedding.shape:", v_embedding.shape)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)

        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        print("qr.shape:", qr.shape)
        print("kr.shape:", kr.shape)
        print("qk.shape:", qk.shape)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        print("stacked_similarity.shape:", stacked_similarity.shape)

        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W * L, 3, self.groups, H, H).sum(dim=1)
        print("stacked_similarity.shape:", stacked_similarity.shape)

        # stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        print("similarity.shape:", similarity.shape)

        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        print("sv.shape:", sv.shape)

        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        print("sve.shape:", sve.shape)

        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W*L, self.out_planes * 2, H)
        print("stacked_similarity.shape:", stacked_similarity.shape)

        output = self.bn_output(stacked_output).view(N, W, L, self.out_planes, 2, H).sum(dim=-2)

        if self.width == 0:  # length
            output = output.permute(0, 3, 1, 2, 4)  # N, L, H ,C, W
        elif self.width == 1:  # height
            output = output.permute(0, 3, 2, 4, 1)  # N, W, L, C, H
        else:  # width
            output = output.permute(0, 3, 4, 1, 2)  # N, H, W, C, L


        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        # nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))
class AxialBlock(nn.Module):
    expansion = 2
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,#8
                 base_width=64, dilation=1, norm_layer=None, kernel_size=(8,16,32)):
        super(AxialBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.))   #width = planes
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.length_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size,width=0)
        self.hight_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size,width=1)
        self.width_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=2)
        self.conv_up = conv1x1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print(out.shape)
        out =  self.length_block(out)
        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResAxialAttentionUNet(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 4
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440

    def __init__(self, layers, input_channels,  num_classes, deep_supervision=False,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.125, img_size=(128,128,128)):
        super(ResAxialAttentionUNet, self).__init__()
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer
        self.inplanes = int(64 * s)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv3d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.conv2 = nn.Conv3d(self.inplanes, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv3d(128, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = norm_layer(128)
        self.bn3 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        block = AxialBlock
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(128 * s), layers[0], kernel_size=(img_size[0] // 2,img_size[1] // 2,img_size[2] // 2))
        self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=(img_size[0] // 2,img_size[1] // 2,img_size[2] // 2),
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=(img_size[0] // 4,img_size[1] // 4,img_size[2] // 4),
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, int(1024 * s), layers[3], stride=2, kernel_size=(img_size[0] // 8,img_size[1] // 8,img_size[2] // 8),
                                       dilate=replace_stride_with_dilation[2])

        # Decoder
        self.decoder1 = nn.Conv3d(int(1024 * 2 * s), int(1024 * 2 * s), kernel_size=3, stride=2, padding=1)
        self.decoder2 = nn.Conv3d(int(1024 * 2 * s), int(1024 * s), kernel_size=3, stride=1, padding=1)
        self.decoder3 = nn.Conv3d(int(1024 * s), int(512 * s), kernel_size=3, stride=1, padding=1)
        self.decoder4 = nn.Conv3d(int(512 * s), int(256 * s), kernel_size=3, stride=1, padding=1)
        self.decoder5 = nn.Conv3d(int(256 * s), int(128 * s), kernel_size=3, stride=1, padding=1)
        self.adjust = nn.Conv3d(int(128 * s), num_classes, kernel_size=1, stride=1, padding=0)
        self.soft = nn.Softmax(dim=1)

    def _make_layer(self, block, planes, blocks, kernel_size=(8,16,32), stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation,
                            norm_layer=norm_layer, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = (kernel_size[0] // 2,kernel_size[1] // 2,kernel_size[2] // 2)


        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        # AxialAttention Encoder
        # pdb.set_trace()
        print("x.shape:",x.shape)

        x = self.conv1(x)
        print("x1.shape:",x.shape)

        x = self.bn1(x)
        print("x2.shape:",x.shape)

        x = self.relu(x)
        print("x3.shape:",x.shape)

        x = self.conv2(x)
        print("x4.shape:",x.shape)

        x = self.bn2(x)
        print("x5.shape:",x.shape)

        x = self.relu(x)
        print("x6.shape:",x.shape)

        x = self.conv3(x)
        print("x7.shape:",x.shape)

        x = self.bn3(x)
        print("x8.shape:",x.shape)

        x = self.relu(x)
        print("x9.shape:",x.shape)


        x1 = self.layer1(x)
        print("x11.shape:",x.shape)


        x2 = self.layer2(x1)
        print("x12.shape:",x.shape)

        # print(x2.shape)
        x3 = self.layer3(x2)
        print("x13.shape:",x.shape)

        # print(x3.shape)
        x4 = self.layer4(x3)
        print("x14.shape:",x.shape)


        x = F.relu(F.interpolate(self.decoder1(x4), scale_factor=(2, 2, 2), mode='trilinear'))
        x = torch.add(x, x4)
        x = F.relu(F.interpolate(self.decoder2(x), scale_factor=(2, 2, 2), mode='trilinear'))
        x = torch.add(x, x3)
        x = F.relu(F.interpolate(self.decoder3(x), scale_factor=(2, 2, 2), mode='trilinear'))
        x = torch.add(x, x2)
        x = F.relu(F.interpolate(self.decoder4(x), scale_factor=(2, 2, 2), mode='trilinear'))
        x = torch.add(x, x1)
        x = F.relu(F.interpolate(self.decoder5(x), scale_factor=(2, 2, 2), mode='trilinear'))
        x = self.adjust(F.relu(x))
        # pdb.set_trace()
        #  out = self.active(out)
        seg_outputs = []
        seg_outputs.append(x)

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]])
        else:
            return seg_outputs[-1]

    def forward(self, x):
        return self._forward_impl(x)