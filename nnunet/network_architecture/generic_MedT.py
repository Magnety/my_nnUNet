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
    def __init__(self, in_planes, out_planes, groups=4, kernel_size=(8, 16, 32),
                 stride=(1, 1, 1), bias=False, width=0):
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
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size[2 - self.width] * 2 - 1),
                                     requires_grad=True)
        query_index = torch.arange(kernel_size[2 - self.width]).unsqueeze(0)
        key_index = torch.arange(kernel_size[2 - self.width]).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size[2 - self.width] - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride != (1, 1, 1):
            self.pooling = nn.AvgPool3d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        # pdb.set_trace()
        # N C L H W
        # ##print("x.shape:",x.shape)
        if self.width == 0:  # length
            x = x.permute(0, 2, 3, 1, 4)  # N, L, H ,C, W
        elif self.width == 1:  # height
            x = x.permute(0, 4, 2, 1, 3)  # N, W, L, C, H
        else:  # width
            x = x.permute(0, 3, 4, 1, 2)  # N, H, W, C, L

        N, W, L, C, H = x.shape
        x = x.contiguous().view(N * W * L, C, H)
        # ##print("x.shape:",x.shape)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        # ##print("qkv.shape:",qkv.shape)

        q, k, v = torch.split(qkv.reshape(N * W * L, self.groups, self.group_planes * 2, H),
                              [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)
        # ##print("q.shape:",q.shape)
        # ##print("k.shape:",k.shape)
        # ##print("v.shape:",v.shape)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2,
                                                                                       self.kernel_size[2 - self.width],
                                                                                       self.kernel_size[2 - self.width])

        # ##print("all_beddings.shape:",all_embeddings.shape)

        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                            [self.group_planes // 2, self.group_planes // 2,
                                                             self.group_planes], dim=0)
        # ##print("q_bedding.shape:", q_embedding.shape)
        # ##print("k_bedding.shape:", k_embedding.shape)
        # ##print("v_bedding.shape:", v_embedding.shape)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)

        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        # ##print("qr.shape:", qr.shape)
        # ##print("kr.shape:", kr.shape)
        # ##print("qk.shape:", qk.shape)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        # ##print("stacked_similarity.shape:", stacked_similarity.shape)

        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W * L, 3, self.groups, H, H)
        # ##print("stacked_similarity.shape:", stacked_similarity.shape)
        stacked_similarity = stacked_similarity.sum(dim=1)
        # ##print("stacked_similarity.shape:", stacked_similarity.shape)

        # stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        # ##print("similarity.shape:", similarity.shape)

        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        # ##print("sv.shape:", sv.shape)

        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        # ##print("sve.shape:", sve.shape)

        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W * L, self.out_planes * 2, H)
        # ##print("stacked_similarity.shape:", stacked_output.shape)

        output = self.bn_output(stacked_output).view(N, W, L, self.out_planes, 2, H).sum(dim=-2)
        # ##print("output.shape:", output.shape)
        if self.width == 0:  # length
            output = output.permute(0, 3, 1, 2, 4)  # N, L, H ,C, W
        elif self.width == 1:  # height
            output = output.permute(0, 3, 2, 4, 1)  # N, W, L, C, H
        else:  # width
            output = output.permute(0, 3, 4, 1, 2)  # N, H, W, C, L

        # ##print("output.shape:", output.shape)

        if self.stride != (1, 1, 1):
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        # nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))


class AxialAttention_dynamic(nn.Module):
    def __init__(self, in_planes, out_planes, groups=4, kernel_size=(8, 16, 32),
                 stride=(1, 1, 1), bias=False, width=0):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention_dynamic, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Priority on encoding

        ## Initial values

        self.f_qr = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.f_kr = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.f_sve = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.f_sv = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size[2 - self.width] * 2 - 1),
                                     requires_grad=True)
        query_index = torch.arange(kernel_size[2 - self.width]).unsqueeze(0)
        key_index = torch.arange(kernel_size[2 - self.width]).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size[2 - self.width] - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride != (1, 1, 1):
            self.pooling = nn.AvgPool3d(stride, stride=stride)

        self.reset_parameters()
        # self.print_para()

    def forward(self, x):
        # ##print("x.shape:",x.shape)
        if self.width == 0:  # length
            x = x.permute(0, 2, 3, 1, 4)  # N, L, H ,C, W
        elif self.width == 1:  # height
            x = x.permute(0, 4, 2, 1, 3)  # N, W, L, C, H
        else:  # width
            x = x.permute(0, 3, 4, 1, 2)  # N, H, W, C, L
        # ##print("x.shape:",x.shape)
        N, W, L, C, H = x.shape
        x = x.contiguous().view(N * W * L, C, H)
        # ##print("x.shape:",x.shape)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        # ##print("qkv.shape:",qkv.shape)

        q, k, v = torch.split(qkv.reshape(N * W * L, self.groups, self.group_planes * 2, H),
                              [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)
        # ##print("q.shape:",q.shape)
        # ##print("k.shape:",k.shape)
        # ##print("v.shape:",v.shape)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2,
                                                                                       self.kernel_size[2 - self.width],
                                                                                       self.kernel_size[2 - self.width])

        # ##print("all_beddings.shape:",all_embeddings.shape)

        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                            [self.group_planes // 2, self.group_planes // 2,
                                                             self.group_planes], dim=0)
        # ##print("q_bedding.shape:", q_embedding.shape)
        # ##print("k_bedding.shape:", k_embedding.shape)
        # ##print("v_bedding.shape:", v_embedding.shape)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        # ##print("qr.shape:", qr.shape)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        # ##print("kr.shape:", kr.shape)

        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        # ##print("qk.shape:", qk.shape)

        # multiply by factors
        qr = torch.mul(qr, self.f_qr)
        # ##print("qr.shape:", qr.shape)

        kr = torch.mul(kr, self.f_kr)
        # ##print("kr.shape:", kr.shape)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        # ##print("stacked_similarity.shape:", stacked_similarity.shape)

        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W * L, 3, self.groups, H, H).sum(dim=1)
        # stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # ##print("stacked_similarity.shape:", stacked_similarity.shape)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        # ##print("similarity.shape:", similarity.shape)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        # ##print("sv.shape:", sv.shape)

        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        # ##print("sve.shape:", sve.shape)

        # multiply by factors
        sv = torch.mul(sv, self.f_sv)
        # ##print("sv.shape:", sv.shape)

        sve = torch.mul(sve, self.f_sve)
        # ##print("sve.shape:", sve.shape)

        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W * L, self.out_planes * 2, H)
        # ##print("stacked_output.shape:", stacked_output.shape)

        output = self.bn_output(stacked_output).view(N, W, L, self.out_planes, 2, H).sum(dim=-2)
        # ##print("output.shape:", output.shape)

        if self.width == 0:  # length
            output = output.permute(0, 3, 1, 2, 4)  # N, L, H ,C, W
        elif self.width == 1:  # height
            output = output.permute(0, 3, 2, 4, 1)  # N, W, L, C, H
        else:  # width
            output = output.permute(0, 3, 4, 1, 2)  # N, H, W, C, L

        # ##print("output.shape:", output.shape)

        if self.stride != (1, 1, 1):
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        # nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))


class AxialAttention_wopos(nn.Module):
    def __init__(self, in_planes, out_planes, groups=4, kernel_size=(8, 16, 32),
                 stride=(1, 1, 1), bias=False, width=0):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention_wopos, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups)

        self.bn_output = nn.BatchNorm1d(out_planes * 1)

        if stride != (1, 1, 1):
            self.pooling = nn.AvgPool3d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        # ##print("aaw_x.shape",x.shape)
        if self.width == 0:  # length
            x = x.permute(0, 2, 3, 1, 4)  # N, L, H ,C, W
        elif self.width == 1:  # height
            x = x.permute(0, 4, 2, 1, 3)  # N, W, L, C, H
        else:  # width
            x = x.permute(0, 3, 4, 1, 2)  # N, H, W, C, L
        # ##print("aaw_x.shape",x.shape)

        N, W, L, C, H = x.shape
        x = x.contiguous().view(N * W * L, C, H)
        # ##print("aaw_x.shape",x.shape)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        # ##print("aaw_qkv.shape:",qkv.shape)

        q, k, v = torch.split(qkv.reshape(N * W * L, self.groups, self.group_planes * 2, H),
                              [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)
        # ##print("aaw_q.shape",q.shape)
        # ##print("aaw_k.shape",k.shape)
        # ##print("aaw_v.shape",v.shape)

        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        # ##print("aaw_qk.shape",qk.shape)

        stacked_similarity = self.bn_similarity(qk).reshape(N * W * L, 1, self.groups, H, H).sum(dim=1).contiguous()
        # ##print("aaw_stacked_similarity.shape",stacked_similarity.shape)

        similarity = F.softmax(stacked_similarity, dim=3)
        # ##print("aaw_similarity.shape",similarity.shape)

        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        # ##print("aaw_sv.shape",sv.shape)

        sv = sv.reshape(N * W * L, self.out_planes * 1, H).contiguous()
        # ##print("aaw_sv.shape",sv.shape)

        output = self.bn_output(sv).reshape(N, W, L, self.out_planes, 1, H).sum(dim=-2).contiguous()
        # ##print("aaw_output.shape",output.shape)

        if self.width == 0:  # length
            output = output.permute(0, 3, 1, 2, 4)  # N, L, H ,C, W
        elif self.width == 1:  # height
            output = output.permute(0, 3, 2, 4, 1)  # N, W, L, C, H
        else:  # width
            output = output.permute(0, 3, 4, 1, 2)  # N, H, W, C, L
        if self.stride != (1, 1, 1):
            output = self.pooling(output)
        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        # nn.init.uniform_(self.relative, -0.1, 0.1)
        # nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))


class AxialBlock(nn.Module):
    expansion = 2

    def __init__(self, conv_op, norm_op, nonlin, inplanes, planes, stride=(1, 1, 1), downsample=None, groups=1,  # 8
                 base_width=64, dilation=1, norm_layer=None, kernel_size=(8, 16, 32)):
        super(AxialBlock, self).__init__()

        if norm_layer is None:
            norm_layer = norm_op
        width = int(planes * (base_width / 64.))  # width = planes
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.length_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size, width=0)
        self.hight_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size, width=1)
        self.width_block = AxialAttention(width, width, groups=groups, stride=stride, kernel_size=kernel_size, width=2)
        self.conv_up = conv1x1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nonlin()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)
        # ##print(out.shape)
        out = self.length_block(out)
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


class AxialBlock_dynamic(nn.Module):
    expansion = 2

    def __init__(self, conv_op, norm_op, nonlin, inplanes, planes, stride=(1, 1, 1), downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=(128, 128, 128)):
        super(AxialBlock_dynamic, self).__init__()
        if norm_layer is None:
            norm_layer = norm_op
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.length_block = AxialAttention_dynamic(width, width, groups=groups, kernel_size=kernel_size, width=0)
        self.hight_block = AxialAttention_dynamic(width, width, groups=groups, kernel_size=kernel_size, width=1)
        self.width_block = AxialAttention_dynamic(width, width, groups=groups, stride=stride, kernel_size=kernel_size,
                                                  width=2)
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
        out = self.length_block(out)
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


class AxialBlock_wopos(nn.Module):
    expansion = 2

    def __init__(self, conv_op, norm_op, nonlin, inplanes, planes, stride=(1, 1, 1), downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=(128, 128, 128)):
        super(AxialBlock_wopos, self).__init__()
        if norm_layer is None:
            norm_layer = norm_op
        # ##print(kernel_size)
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1x1(inplanes, width)
        self.conv1 = conv_op(width, width, kernel_size=1)
        self.bn1 = norm_layer(width)
        self.length_block = AxialAttention_wopos(width, width, groups=groups, kernel_size=kernel_size, width=0)
        self.hight_block = AxialAttention_wopos(width, width, groups=groups, kernel_size=kernel_size, width=1)
        self.width_block = AxialAttention_wopos(width, width, groups=groups, stride=stride, kernel_size=kernel_size,
                                                width=2)
        self.conv_up = conv1x1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # pdb.set_trace()

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)
        # ##print("aaw_out.shape",out.shape)
        out = self.length_block(out)
        # ##print("aaw_out.shape",out.shape)
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
    DEFAULT_BATCH_SIZE_3D = 2
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

    def __init__(self, layers, input_channels, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None,
                 seg_output_use_bias=False, deep_supervision=True,
                 groups=2, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.25, img_size=(128, 128, 128), ):
        super(ResAxialAttentionUNet, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling  # True
        self.convolutional_pooling = convolutional_pooling  # True
        self.upscale_logits = upscale_logits  # False
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}  #
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}
        self.nonlin = nonlin  # nn.LeakyReLU
        self.nonlin_kwargs = nonlin_kwargs  # {'negative_slope': 1e-2, 'inplace': True}
        self.dropout_op_kwargs = dropout_op_kwargs  # {'p': 0, 'inplace': True}
        self.norm_op_kwargs = norm_op_kwargs  # {'eps': 1e-5, 'affine': True}
        self.weightInitializer = weightInitializer  # InitWeights_He(1e-2)
        self.conv_op = conv_op  # nn.Conv3d
        self.norm_op = norm_op  # nn.InstanceNorm3d
        self.dropout_op = dropout_op  # nn.Dropout3d
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        if conv_op == nn.Conv2d:
            # pool_op = nn.MaxPool2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d:
            pool_op = nn.AvgPool3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features
        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)

        if norm_layer is None:
            norm_layer = self.norm_op
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
        self.conv1 = self.conv_op(input_channels, self.inplanes, kernel_size=7, stride=(1, 2, 2), padding=3, bias=False)
        self.conv2 = self.conv_op(self.inplanes, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = self.conv_op(64, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = norm_layer(64)
        self.bn3 = norm_layer(self.inplanes)
        self.relu = self.nonlin()
        block = AxialBlock
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.conv_op, self.norm_op, self.nonlin, block, int(64 * s), layers[0],
                                       kernel_size=(img_size[0], img_size[1] // 2, img_size[2] // 2))
        self.layer2 = self._make_layer(self.conv_op, self.norm_op, self.nonlin, block, int(128 * s), layers[1],
                                       stride=(1, 2, 2), kernel_size=(img_size[0], img_size[1] // 2, img_size[2] // 2),
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(self.conv_op, self.norm_op, self.nonlin, block, int(256 * s), layers[2],
                                       stride=(1, 2, 2), kernel_size=(img_size[0], img_size[1] // 4, img_size[2] // 4),
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(self.conv_op, self.norm_op, self.nonlin, block, int(512 * s), layers[3],
                                       stride=(1, 2, 2), kernel_size=(img_size[0], img_size[1] // 8, img_size[2] // 8),
                                       dilate=replace_stride_with_dilation[2])

        # Decoder
        self.decoder1 = self.conv_op(int(512 * 2 * s), int(512 * 2 * s), kernel_size=3, stride=2, padding=1)
        self.decoder2 = self.conv_op(int(512 * 2 * s), int(512 * s), kernel_size=3, stride=1, padding=1)
        self.decoder3 = self.conv_op(int(512 * s), int(256 * s), kernel_size=3, stride=1, padding=1)
        self.decoder4 = self.conv_op(int(256 * s), int(128 * s), kernel_size=3, stride=1, padding=1)
        self.decoder5 = self.conv_op(int(128 * s), int(64 * s), kernel_size=3, stride=1, padding=1)
        self.adjust = self.conv_op(int(64 * s), num_classes, kernel_size=1, stride=1, padding=0)
        self.soft = nn.Softmax(dim=1)

    def _make_layer(self, conv_op, norm_op, nonlin, block, planes, blocks, kernel_size=(8, 16, 32), stride=(1, 1, 1),
                    dilate=False):
        norm_layer = norm_op
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != (1, 1, 1) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(conv_op, norm_op, nonlin, self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation,
                            norm_layer=norm_layer, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        if stride != (1, 1, 1):
            kernel_size = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)

        for _ in range(1, blocks):
            layers.append(block(conv_op, norm_op, nonlin, self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        # AxialAttention Encoder
        # pdb.set_trace()
        # ##print("x.shape:",x.shape)

        x = self.conv1(x)
        # ##print("x1.shape:",x.shape)

        x = self.bn1(x)
        # ##print("x2.shape:",x.shape)

        x = self.relu(x)
        # ##print("x3.shape:",x.shape)

        x = self.conv2(x)
        # ##print("x4.shape:",x.shape)

        x = self.bn2(x)
        # ##print("x5.shape:",x.shape)

        x = self.relu(x)
        # ##print("x6.shape:",x.shape)

        x = self.conv3(x)
        # ##print("x7.shape:",x.shape)

        x = self.bn3(x)
        # ##print("x8.shape:",x.shape)

        x = self.relu(x)
        # ##print("x9.shape:",x.shape)

        x1 = self.layer1(x)
        # ##print("x11.shape:",x1.shape)

        x2 = self.layer2(x1)
        # ##print("x12.shape:",x2.shape)

        # ##print(x2.shape)
        x3 = self.layer3(x2)
        # ##print("x13.shape:",x3.shape)

        # ##print(x3.shape)
        x4 = self.layer4(x3)
        # ##print("x14.shape:",x4.shape)

        x = F.relu(F.interpolate(self.decoder1(x4), scale_factor=(2, 2, 2), mode='trilinear'))
        x = torch.add(x, x4)
        x = F.relu(F.interpolate(self.decoder2(x), scale_factor=(1, 2, 2), mode='trilinear'))
        x = torch.add(x, x3)
        x = F.relu(F.interpolate(self.decoder3(x), scale_factor=(1, 2, 2), mode='trilinear'))
        x = torch.add(x, x2)
        x = F.relu(F.interpolate(self.decoder4(x), scale_factor=(1, 2, 2), mode='trilinear'))
        x = torch.add(x, x1)
        x = F.relu(F.interpolate(self.decoder5(x), scale_factor=(1, 2, 2), mode='trilinear'))
        x = self.adjust(F.relu(x))
        # pdb.set_trace()
        #  out = self.active(out)
        return x

    def forward(self, x):
        seg_outputs = []
        seg_outputs.append(self._forward_impl(x))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]])
        else:
            return seg_outputs[-1]

    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param deep_supervision:
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :param num_modalities:
        :param num_classes:
        :param pool_op_kernel_sizes:
        :return:
        """
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = 10  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
            # ##print(p, map_size, num_feat, tmp)
        return tmp


class medt_net(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2
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

    def __init__(self, layers, input_channels, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None,
                 seg_output_use_bias=False, deep_supervision=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.25, img_size=(128, 128, 128), ):
        super(medt_net, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling  # True
        self.convolutional_pooling = convolutional_pooling  # True
        self.upscale_logits = upscale_logits  # False
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}  #
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}
        self.nonlin = nonlin  # nn.LeakyReLU
        self.nonlin_kwargs = nonlin_kwargs  # {'negative_slope': 1e-2, 'inplace': True}
        self.dropout_op_kwargs = dropout_op_kwargs  # {'p': 0, 'inplace': True}
        self.norm_op_kwargs = norm_op_kwargs  # {'eps': 1e-5, 'affine': True}
        self.weightInitializer = weightInitializer  # InitWeights_He(1e-2)
        self.conv_op = conv_op  # nn.Conv3d
        self.norm_op = norm_op  # nn.InstanceNorm3d
        self.dropout_op = dropout_op  # nn.Dropout3d
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        if conv_op == nn.Conv2d:
            # pool_op = nn.MaxPool2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d:
            pool_op = nn.AvgPool3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features
        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
        block = AxialBlock_dynamic
        block_2 = AxialBlock_wopos
        if norm_layer is None:
            norm_layer = self.norm_op
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
        self.conv1 = self.conv_op(input_channels, self.inplanes, kernel_size=7, stride=(1, 2, 2), padding=3,
                                  bias=False)
        self.conv2 = self.conv_op(self.inplanes, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = self.conv_op(128, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.bn2 = norm_layer(128)
        self.bn3 = norm_layer(self.inplanes)
        # self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.conv_op, self.norm_op, self.nonlin, block, int(128 * s), layers[0],
                                       kernel_size=(img_size[0], img_size[1] // 2, img_size[2] // 2))
        self.layer2 = self._make_layer(self.conv_op, self.norm_op, self.nonlin, block, int(256 * s), layers[1],
                                       stride=(1, 2, 2), kernel_size=(img_size[0], img_size[1] // 2, img_size[2] // 2),
                                       dilate=replace_stride_with_dilation[0])
        # self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=(img_size//4),
        #                                dilate=replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer(block, int(1024 * s), layers[3], stride=2, kernel_size=(img_size//8),
        #                                dilate=replace_stride_with_dilation[2])

        # Decoder
        # self.decoder1 = nn.Conv2d(int(1024 *2*s)      ,        int(1024*2*s), kernel_size=3, stride=2, padding=1)
        # self.decoder2 = nn.Conv2d(int(1024  *2*s)     , int(1024*s), kernel_size=3, stride=1, padding=1)
        # self.decoder3 = nn.Conv2d(int(1024*s),  int(512*s), kernel_size=3, stride=1, padding=1)
        self.decoder4 = self.conv_op(int(512 * s), int(256 * s), kernel_size=3, stride=1, padding=1)
        self.decoder5 = self.conv_op(int(256 * s), int(128 * s), kernel_size=3, stride=1, padding=1)
        self.adjust = self.conv_op(int(128 * s), num_classes, kernel_size=1, stride=1, padding=0)
        self.soft = nn.Softmax(dim=1)

        self.conv1_p = self.conv_op(input_channels, self.inplanes, kernel_size=7, stride=(1, 2, 2), padding=3,
                                    bias=False)
        self.conv2_p = self.conv_op(self.inplanes, 128, kernel_size=3, stride=1, padding=1,
                                    bias=False)
        self.conv3_p = self.conv_op(128, self.inplanes, kernel_size=3, stride=1, padding=1,
                                    bias=False)
        # self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_p = norm_layer(self.inplanes)
        self.bn2_p = norm_layer(128)
        self.bn3_p = norm_layer(self.inplanes)

        self.relu_p = self.nonlin

        img_size_p = (img_size[0], img_size[1] // 4, img_size[2] // 4)
        self.layer1_p = self._make_layer(self.conv_op, self.norm_op, self.nonlin, block_2, int(128 * s), layers[0],
                                         kernel_size=(img_size_p[0], img_size_p[1] // 2, img_size_p[2] // 2))
        self.layer2_p = self._make_layer(self.conv_op, self.norm_op, self.nonlin, block_2, int(256 * s), layers[1],
                                         stride=(1, 2, 2),
                                         kernel_size=(img_size_p[0], img_size_p[1] // 2, img_size_p[2] // 2),
                                         dilate=replace_stride_with_dilation[0])
        self.layer3_p = self._make_layer(self.conv_op, self.norm_op, self.nonlin, block_2, int(512 * s), layers[2],
                                         stride=(1, 2, 2),
                                         kernel_size=(img_size_p[0], img_size_p[1] // 4, img_size_p[2] // 4),
                                         dilate=replace_stride_with_dilation[1])
        self.layer4_p = self._make_layer(self.conv_op, self.norm_op, self.nonlin, block_2, int(1024 * s), layers[3],
                                         stride=(1, 2, 2),
                                         kernel_size=(img_size_p[0], img_size_p[1] // 8, img_size_p[2] // 8),
                                         dilate=replace_stride_with_dilation[2])

        # Decoder
        self.decoder1_p = self.conv_op(int(1024 * 2 * s), int(1024 * 2 * s), kernel_size=3, stride=2, padding=1)
        self.decoder2_p = self.conv_op(int(1024 * 2 * s), int(1024 * s), kernel_size=3, stride=1, padding=1)
        self.decoder3_p = self.conv_op(int(1024 * s), int(512 * s), kernel_size=3, stride=1, padding=1)
        self.decoder4_p = self.conv_op(int(512 * s), int(256 * s), kernel_size=3, stride=1, padding=1)
        self.decoder5_p = self.conv_op(int(256 * s), int(128 * s), kernel_size=3, stride=1, padding=1)

        self.decoderf = self.conv_op(int(128 * s), int(128 * s), kernel_size=3, stride=1, padding=1)
        self.adjust_p = self.conv_op(int(128 * s), num_classes, kernel_size=1, stride=1, padding=0)
        self.soft_p = nn.Softmax(dim=1)

    def _make_layer(self, conv_op, norm_op, nonlin, block, planes, blocks, kernel_size=(128, 128, 128),
                    stride=(1, 1, 1), dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != (1, 1, 1) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(conv_op, norm_op, nonlin, self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation,
                            norm_layer=norm_layer, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        if stride != (1, 1, 1):
            kernel_size = (kernel_size[0] // 1, kernel_size[1] // 2, kernel_size[2] // 2)

        for _ in range(1, blocks):
            layers.append(block(conv_op, norm_op, nonlin, self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        xin = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        # x = F.max_pool2d(x,2,2)
        x = self.relu(x)

        # x = self.maxpool(x)
        # pdb.set_trace()
        x1 = self.layer1(x)
        # ##print("x1.shape",x1.shape)
        x2 = self.layer2(x1)
        # ##print("x2.shape",x2.shape)
        # x3 = self.layer3(x2)
        # ###print(x3.shape)
        # x4 = self.layer4(x3)
        # ###print(x4.shape)
        # x = F.relu(F.interpolate(self.decoder1(x4), scale_factor=(2,2), mode ='trilinear'))
        # x = torch.add(x, x4)
        # x = F.relu(F.interpolate(self.decoder2(x4) , scale_factor=(2,2), mode ='trilinear'))
        # x = torch.add(x, x3)
        # x = F.relu(F.interpolate(self.decoder3(x3) , scale_factor=(2,2), mode ='trilinear'))
        # x = torch.add(x, x2)
        x = F.relu(F.interpolate(self.decoder4(x2), scale_factor=(1, 2, 2), mode='trilinear'))
        # ##print("x.shape",x.shape)

        x = torch.add(x, x1)
        x = F.relu(F.interpolate(self.decoder5(x), scale_factor=(1, 2, 2), mode='trilinear'))
        # ##print(x.shape)

        # end of full image training

        # y_out = torch.ones((1,2,128,128))
        x_loc = x.clone()
        # x = F.relu(F.interpolate(self.decoder5(x) , scale_factor=(2,2), mode ='trilinear'))
        # start
        for i in range(0, 4):
            for j in range(0, 4):
                x_p = xin[:, :, :, 32 * i:32 * (i + 1), 32 * j:32 * (j + 1)]
                # begin patch wise
                # ##print("x_p.shape",x_p.shape)
                x_p = self.conv1_p(x_p)
                x_p = self.bn1_p(x_p)
                # x = F.max_pool2d(x,2,2)
                x_p = self.relu(x_p)

                x_p = self.conv2_p(x_p)
                x_p = self.bn2_p(x_p)
                # x = F.max_pool2d(x,2,2)
                x_p = self.relu(x_p)
                x_p = self.conv3_p(x_p)
                x_p = self.bn3_p(x_p)
                # x = F.max_pool2d(x,2,2)
                x_p = self.relu(x_p)

                # x = self.maxpool(x)
                # pdb.set_trace()
                # ##print("xp.shape",x_p.shape)
                x1_p = self.layer1_p(x_p)
                # ##print(x1.shape)
                x2_p = self.layer2_p(x1_p)
                # ##print(x2.shape)
                x3_p = self.layer3_p(x2_p)
                # ###print(x3.shape)
                x4_p = self.layer4_p(x3_p)

                x_p = F.relu(F.interpolate(self.decoder1_p(x4_p), scale_factor=(2, 2, 2), mode='trilinear'))
                x_p = torch.add(x_p, x4_p)
                x_p = F.relu(F.interpolate(self.decoder2_p(x_p), scale_factor=(1, 2, 2), mode='trilinear'))
                x_p = torch.add(x_p, x3_p)
                x_p = F.relu(F.interpolate(self.decoder3_p(x_p), scale_factor=(1, 2, 2), mode='trilinear'))
                x_p = torch.add(x_p, x2_p)
                x_p = F.relu(F.interpolate(self.decoder4_p(x_p), scale_factor=(1, 2, 2), mode='trilinear'))
                x_p = torch.add(x_p, x1_p)
                x_p = F.relu(F.interpolate(self.decoder5_p(x_p), scale_factor=(1, 2, 2), mode='trilinear'))

                x_loc[:, :, :, 32 * i:32 * (i + 1), 32 * j:32 * (j + 1)] = x_p

        x = torch.add(x, x_loc)
        x = F.relu(self.decoderf(x))

        x = self.adjust(F.relu(x))

        # pdb.set_trace()
        return x

    def forward(self, x):
        seg_outputs = []
        seg_outputs.append(self._forward_impl(x))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]])
        else:
            return seg_outputs[-1]


class TUNet(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2
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

    def __init__(self, layers, input_channels, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None,
                 seg_output_use_bias=False, deep_supervision=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.25, img_size=(128, 128, 128), ):
        super(TUNet, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling  # True
        self.convolutional_pooling = convolutional_pooling  # True
        self.upscale_logits = upscale_logits  # False
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}  #
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}
        self.nonlin = nonlin  # nn.LeakyReLU
        self.nonlin_kwargs = nonlin_kwargs  # {'negative_slope': 1e-2, 'inplace': True}
        self.dropout_op_kwargs = dropout_op_kwargs  # {'p': 0, 'inplace': True}
        self.norm_op_kwargs = norm_op_kwargs  # {'eps': 1e-5, 'affine': True}
        self.weightInitializer = weightInitializer  # InitWeights_He(1e-2)
        self.conv_op = conv_op  # nn.Conv3d
        self.norm_op = norm_op  # nn.InstanceNorm3d
        self.dropout_op = dropout_op  # nn.Dropout3d
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        if conv_op == nn.Conv2d:
            # pool_op = nn.MaxPool2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d:
            pool_op = nn.AvgPool3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features
        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
        block = AxialBlock_dynamic
        block_2 = AxialBlock_wopos
        if norm_layer is None:
            norm_layer = self.norm_op
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

        self.conv1_u = self.conv_op(input_channels, self.inplanes, kernel_size=7, stride=(2, 2, 2), padding=3,
                                    bias=False)
        self.conv2_u = nn.Sequential(
            self.conv_op(self.inplanes, self.inplanes * 2, kernel_size=3, stride=(1,2,2), padding=1, bias=True),
            nn.GroupNorm(16, self.inplanes * 2),
            nn.PReLU(),
            self.conv_op(self.inplanes * 2, self.inplanes * 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(16, self.inplanes * 2),
            nn.PReLU(),
        )
        self.conv3_u = nn.Sequential(
            self.conv_op(self.inplanes * 2, self.inplanes * 4, kernel_size=3, stride=(1,2,2), padding=1, bias=True),
            nn.GroupNorm(16, self.inplanes * 4),
            nn.PReLU(),
            self.conv_op(self.inplanes * 4, self.inplanes * 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(16, self.inplanes * 4),
            nn.PReLU(),
        )
        self.conv4_u = nn.Sequential(
            self.conv_op(self.inplanes * 4, self.inplanes * 8, kernel_size=3, stride=(1,2,2), padding=1, bias=True),
            nn.GroupNorm(16, self.inplanes * 8),
            nn.PReLU(),
            self.conv_op(self.inplanes * 8, self.inplanes * 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(16, self.inplanes * 8),
            nn.PReLU(),
        )
        self.decoder1_u = self.conv_op(self.inplanes * 8, self.inplanes * 4, kernel_size=3, stride=1, padding=1)
        self.decoder2_u = self.conv_op(self.inplanes * 4, self.inplanes * 2, kernel_size=3, stride=1, padding=1)
        self.decoder3_u = self.conv_op(self.inplanes * 2, self.inplanes * 2, kernel_size=3, stride=1, padding=1)
        # self.decoder4_u = self.conv_op(int(512 * s), int(256 * s), kernel_size=3, stride=1, padding=1)
        # self.decoder5_u = self.conv_op(int(256 * s), int(128 * s), kernel_size=3, stride=1, padding=1)

        self.conv1_p = self.conv_op(input_channels, self.inplanes, kernel_size=7, stride=(1, 2, 2), padding=3,
                                    bias=False)
        self.conv2_p = self.conv_op(self.inplanes, 128, kernel_size=3, stride=1, padding=1,
                                    bias=False)
        self.conv3_p = self.conv_op(128, self.inplanes, kernel_size=3, stride=1, padding=1,
                                    bias=False)
        # self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_p = norm_layer(self.inplanes)
        self.bn2_p = norm_layer(128)
        self.bn3_p = norm_layer(self.inplanes)

        self.relu_p = self.nonlin
        self.relu = nn.ReLU(inplace=True)

        img_size_p = (img_size[0], img_size[1] // 4, img_size[2] // 4)
        self.layer1_p = self._make_layer(self.conv_op, self.norm_op, self.nonlin, block, int(128 * s), layers[0],
                                         kernel_size=(img_size_p[0], img_size_p[1] // 2, img_size_p[2] // 2))
        self.layer2_p = self._make_layer(self.conv_op, self.norm_op, self.nonlin, block, int(256 * s), layers[1],
                                         stride=(1, 2, 2),
                                         kernel_size=(img_size_p[0], img_size_p[1] // 2, img_size_p[2] // 2),
                                         dilate=replace_stride_with_dilation[0])
        self.layer3_p = self._make_layer(self.conv_op, self.norm_op, self.nonlin, block, int(512 * s), layers[2],
                                         stride=(1, 2, 2),
                                         kernel_size=(img_size_p[0], img_size_p[1] // 4, img_size_p[2] // 4),
                                         dilate=replace_stride_with_dilation[1])
        self.layer4_p = self._make_layer(self.conv_op, self.norm_op, self.nonlin, block, int(1024 * s), layers[3],
                                         stride=(1, 2, 2),
                                         kernel_size=(img_size_p[0], img_size_p[1] // 8, img_size_p[2] // 8),
                                         dilate=replace_stride_with_dilation[2])
        # Decoder
        self.decoder1_p = self.conv_op(int(1024 * 2 * s), int(1024 * 2 * s), kernel_size=3, stride=2, padding=1)
        self.decoder2_p = self.conv_op(int(1024 * 2 * s), int(1024 * s), kernel_size=3, stride=1, padding=1)
        self.decoder3_p = self.conv_op(int(1024 * s), int(512 * s), kernel_size=3, stride=1, padding=1)
        self.decoder4_p = self.conv_op(int(512 * s), int(256 * s), kernel_size=3, stride=1, padding=1)
        self.decoder5_p = self.conv_op(int(256 * s), int(128 * s), kernel_size=3, stride=1, padding=1)

        self.decoderf = self.conv_op(int(128 * s), int(128 * s), kernel_size=3, stride=1, padding=1)
        self.adjust_p = self.conv_op(int(128 * s), num_classes, kernel_size=1, stride=1, padding=0)
        self.soft_p = nn.Softmax(dim=1)

    def _make_layer(self, conv_op, norm_op, nonlin, block, planes, blocks, kernel_size=(128, 128, 128),
                    stride=(1, 1, 1), dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != (1, 1, 1) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(conv_op, norm_op, nonlin, self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation,
                            norm_layer=norm_layer, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        if stride != (1, 1, 1):
            kernel_size = (kernel_size[0] // 1, kernel_size[1] // 2, kernel_size[2] // 2)

        for _ in range(1, blocks):
            layers.append(block(conv_op, norm_op, nonlin, self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        xin = x.clone()
        x = self.conv1_u(x)
        ##print(x.shape)
        x = self.conv2_u(x)
        ##print(x.shape)
        x = self.conv3_u(x)
        ##print(x.shape)
        x = self.conv4_u(x)
        x = F.relu(F.interpolate(x, scale_factor=(1, 2, 2), mode='trilinear',align_corners=True))
        ##print("de",x.shape)

        x = F.relu(F.interpolate(self.decoder1_u(x), scale_factor=(1, 2, 2), mode='trilinear',align_corners=True))
        ##print("de",x.shape)

        x = F.relu(F.interpolate(self.decoder2_u(x), scale_factor=(1, 2, 2), mode='trilinear',align_corners=True))
        ##print("de",x.shape)

        x = F.relu(F.interpolate(self.decoder3_u(x), scale_factor=(2, 2, 2), mode='trilinear',align_corners=True))
        ##print("de",x.shape)

        x_loc = x.clone()
        # x = F.relu(F.interpolate(self.decoder5(x) , scale_factor=(2,2), mode ='trilinear'))
        # start
        for i in range(0, 4):
            for j in range(0, 4):
                x_p = xin[:, :, :, 32 * i:32 * (i + 1), 32 * j:32 * (j + 1)]
                # begin patch wise
                # ##print("x_p.shape",x_p.shape)
                x_p = self.conv1_p(x_p)
                x_p = self.bn1_p(x_p)
                # x = F.max_pool2d(x,2,2)
                x_p = self.relu(x_p)

                x_p = self.conv2_p(x_p)
                x_p = self.bn2_p(x_p)
                # x = F.max_pool2d(x,2,2)
                x_p = self.relu(x_p)
                x_p = self.conv3_p(x_p)
                x_p = self.bn3_p(x_p)
                # x = F.max_pool2d(x,2,2)
                x_p = self.relu(x_p)

                # x = self.maxpool(x)
                # pdb.set_trace()
                # ##print("xp.shape",x_p.shape)
                x1_p = self.layer1_p(x_p)
                ##print("layer1.shape",x1_p.shape)
                # ##print(x1.shape)
                x2_p = self.layer2_p(x1_p)
                ##print("layer2.shape",x2_p.shape)

                # ##print(x2.shape)
                x3_p = self.layer3_p(x2_p)
                ##print("layer3.shape",x3_p.shape)

                # ###print(x3.shape)
                x4_p = self.layer4_p(x3_p)
                ##print("layer4.shape",x4_p.shape)

                x_p = F.relu(F.interpolate(self.decoder1_p(x4_p), scale_factor=(2, 2, 2), mode='trilinear',align_corners=True))
                ##print("_layer1.shape",x_p.shape)

                x_p = torch.add(x_p, x4_p)
                x_p = F.relu(F.interpolate(self.decoder2_p(x_p), scale_factor=(1, 2, 2), mode='trilinear',align_corners=True))
                ##print("_layer2.shape",x_p.shape)

                x_p = torch.add(x_p, x3_p)
                x_p = F.relu(F.interpolate(self.decoder3_p(x_p), scale_factor=(1, 2, 2), mode='trilinear',align_corners=True))
                ##print("_layer3.shape",x_p.shape)

                x_p = torch.add(x_p, x2_p)
                x_p = F.relu(F.interpolate(self.decoder4_p(x_p), scale_factor=(1, 2, 2), mode='trilinear',align_corners=True))
                ##print("_layer4.shape",x_p.shape)

                x_p = torch.add(x_p, x1_p)
                x_p = F.relu(F.interpolate(self.decoder5_p(x_p), scale_factor=(1, 2, 2), mode='trilinear',align_corners=True))
                ##print("_layer5.shape",x_p.shape)


                x_loc[:, :, :, 32 * i:32 * (i + 1), 32 * j:32 * (j + 1)] = x_p

        x = torch.add(x, x_loc)
        x = F.relu(self.decoderf(x))
        x = self.adjust_p(F.relu(x))

        # pdb.set_trace()
        return x

    def forward(self, x):
        seg_outputs = []
        seg_outputs.append(self._forward_impl(x))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]])
        else:
            return seg_outputs[-1]