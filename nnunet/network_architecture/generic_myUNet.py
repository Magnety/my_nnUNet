#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch.nn.functional as F
from copy import deepcopy

from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional
class ConvDropoutNormNonlin(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class ConvDropoutNonlinNorm(ConvDropoutNormNonlin):
    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.instnorm(self.lrelu(x))

def passthrough(x, **kwargs):
    return x
class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        #super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)



def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

class InputTransition(nn.Module):
    def __init__(self, input_channels, output_channels,conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(InputTransition, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}
        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.conv1 = self.conv_op(input_channels, output_channels, kernel_size=5, padding=2)
        self.bn1 = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.relu1 = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x))
        # split input in to 16 channels
        x12 = torch.cat((x, x, x, x, x, x, x, x,x, x, x,x),1)
        out = self.relu1(torch.add(out, x12))
        return out

class DownTransition(nn.Module):
    def __init__(self, input_features,num_conv,first_stride,conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None,basic_block=ConvDropoutNormNonlin,dropout=False):
        super(DownTransition, self).__init__()
        output_features= 2*input_features
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        conv_kwargs = {'kernel_size': 5, 'padding': 2, 'bias': True}
        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout = dropout

        self.down_conv = self.conv_op(input_features, output_features, kernel_size=2, stride=first_stride)
        self.bn1 = self.norm_op(output_features, **self.norm_op_kwargs)
        self.relu1 = self.nonlin(**self.nonlin_kwargs)
        self.relu2 = self.nonlin(**self.nonlin_kwargs)
        self.do1 = passthrough
        if self.dropout:
            self.do1 = self.dropout_op(**self.dropout_op_kwargs)
        self.ops = StackedConvLayers(output_features, output_features, num_conv,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride=None, basic_block=basic_block)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, input_features, output_features,num_conv, first_stride, conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, basic_block=ConvDropoutNormNonlin,dropout=False):
        super(UpTransition, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        conv_kwargs = {'kernel_size': 5, 'padding': 2, 'bias': True}
        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout = dropout
        self.up_conv = self.conv_op(input_features, output_features//2, kernel_size=1, stride=1)
        self.bn1 = self.norm_op(output_features//2, **self.norm_op_kwargs)
        self.relu1 = self.nonlin(**self.nonlin_kwargs)
        self.relu2 = self.nonlin(**self.nonlin_kwargs)
        self.do1 = passthrough
        self.do2 = self.dropout_op(**self.dropout_op_kwargs)
        if self.dropout:
            self.do1 = self.dropout_op(**self.dropout_op_kwargs)


        self.ops = StackedConvLayers(output_features, output_features, num_conv,
                                     self.conv_op, self.conv_kwargs, self.norm_op,
                                     self.norm_op_kwargs, self.dropout_op,
                                     self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                     first_stride=None, basic_block=basic_block)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class ReconsTransition(nn.Module):
    def __init__(self, input_features,output_features,conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None ):
        super(ReconsTransition, self).__init__()

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}
        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.conv1 = self.conv_op(input_features,input_features, kernel_size=5, padding=2)
        self.bn1 = self.norm_op(input_features, **self.norm_op_kwargs)
        self.conv2 =self.conv_op(input_features,output_features, kernel_size=1)
        self.relu1 = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        # # make channels the last axis
        # out = out.permute(0, 2, 3, 4, 1).contiguous()
        # # flatten
        # out = out.view(out.numel() // 2, 2)
        # out = self.softmax(out)
        # # treat channel 0 as the predicted output
        return out


class OutputTransition(nn.Module):
    def __init__(self, input_features, output_features, conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(OutputTransition, self).__init__()

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}
        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.conv1 = self.conv_op(input_features, output_features, kernel_size=5, padding=2)
        self.bn1 = self.norm_op(output_features, **self.norm_op_kwargs)
        self.conv2 = self.conv_op(output_features, output_features, kernel_size=1)
        self.relu1 = self.nonlin(**self.nonlin_kwargs)
        self.softmax = F.softmax
    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        # # make channels the last axis
        # out = out.permute(0, 2, 3, 4, 1).contiguous()
        # # flatten
        # out = out.view(out.numel() // 2, 2)
        # out = self.softmax(out)
        # # treat channel 0 as the predicted output
        return out
class Deep_Attention_block(nn.Module):
    """
    Attention Block
    """
    def __init__(self, F_up, F_down, F_out,conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(Deep_Attention_block, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}
        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op




        self.W_g = nn.Sequential(
            self.conv_op(F_up, F_out, kernel_size=1, stride=1, padding=0, bias=True),
            self.norm_op(F_out, **self.norm_op_kwargs)
        )
        self.W_x = nn.Sequential(




            self.conv_op(F_down, F_out, kernel_size=1, stride=1, padding=0, bias=True),
            self.norm_op(F_out, **self.norm_op_kwargs)
        )
        self.fuse = nn.Sequential(


            self.conv_op(360, 3*F_out, kernel_size=1), self.norm_op(3*F_out,**self.norm_op_kwargs), self.nonlin(**self.nonlin_kwargs),
            self.conv_op(3*F_out, 2*F_out, kernel_size=3, padding=1), self.norm_op(2*F_out,**self.norm_op_kwargs), self.nonlin(**self.nonlin_kwargs),
            self.conv_op(2*F_out, F_out, kernel_size=3, padding=1), self.norm_op(F_out,**self.norm_op_kwargs), self.nonlin(**self.nonlin_kwargs),
        )
        self.psi = nn.Sequential(
            self.conv_op(F_out, 1, kernel_size=1, stride=1, padding=0, bias=True),
            self.norm_op(1,**self.norm_op_kwargs),
            nn.Sigmoid()
        )
        self.out_att = nn.Sequential(
            self.conv_op(2*F_out, F_out, kernel_size=1),self.norm_op(F_out,**self.norm_op_kwargs), self.nonlin(**self.nonlin_kwargs),
            self.conv_op(F_out, F_out, kernel_size=3, padding=1),self.norm_op(F_out,**self.norm_op_kwargs), self.nonlin(**self.nonlin_kwargs),
            self.conv_op(F_out, F_out, kernel_size=3, padding=1), self.norm_op(F_out,**self.norm_op_kwargs), self.nonlin(**self.nonlin_kwargs),)
        self.PReLU = self.nonlin(**self.nonlin_kwargs)

    def forward(self, e1,e2,e3,e4,g,x):
        e1_resample = F.upsample(e1,size=x.size()[2:], mode='trilinear')
        e2_resample = F.upsample(e2,size=x.size()[2:], mode='trilinear')
        e3_resample = F.upsample(e3, size=x.size()[2:], mode='trilinear')
        e4_resample = F.upsample(e4, size=x.size()[2:], mode='trilinear')
        fms_concat = torch.cat((e1_resample,e2_resample,e3_resample,e4_resample),1)
        fms_fuse = self.fuse(fms_concat)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.PReLU(g1 + x1)
        psi = self.psi(psi)
        local_att = x * psi
        total_att = torch.cat((fms_fuse,local_att),1)
        out_att = self.out_att(total_att)
        return out_att
class VNet_recons(SegmentationNetwork):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
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

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=False, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False):
        super(VNet_recons, self).__init__()
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
            upsample_mode = 'bilinear'
            pool_op = nn.MaxPool2d
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            transpconv = nn.ConvTranspose3d
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
        channels = 12
        self.in_tr = InputTransition(input_channels=1, output_channels=channels,conv_op=self.conv_op, conv_kwargs=self.conv_kwargs,
                 norm_op=self.norm_op, norm_op_kwargs=self.norm_op_kwargs,
                 dropout_op=self.dropout_op, dropout_op_kwargs=self.dropout_op_kwargs,
                 nonlin=self.nonlin, nonlin_kwargs=self.nonlin_kwargs) #16
        self.down_tr32 = DownTransition(input_features=channels,num_conv=1,first_stride=2,conv_op=self.conv_op, conv_kwargs=self.conv_kwargs,
                 norm_op=self.norm_op, norm_op_kwargs=self.norm_op_kwargs,
                 dropout_op=self.dropout_op, dropout_op_kwargs=self.dropout_op_kwargs,
                 nonlin=self.nonlin, nonlin_kwargs=self.nonlin_kwargs,basic_block=ConvDropoutNormNonlin,dropout=False) #32
        self.down_tr64 = DownTransition(input_features=2*channels,num_conv=2,first_stride=2,conv_op=self.conv_op, conv_kwargs=self.conv_kwargs,
                 norm_op=self.norm_op, norm_op_kwargs=self.norm_op_kwargs,
                 dropout_op=self.dropout_op, dropout_op_kwargs=self.dropout_op_kwargs,
                 nonlin=self.nonlin, nonlin_kwargs=self.nonlin_kwargs,basic_block=ConvDropoutNormNonlin,dropout=False) #64
        self.down_tr128 = DownTransition(input_features=4*channels,num_conv=3,first_stride=2,conv_op=self.conv_op, conv_kwargs=self.conv_kwargs,
                 norm_op=self.norm_op, norm_op_kwargs=self.norm_op_kwargs,
                 dropout_op=self.dropout_op, dropout_op_kwargs=self.dropout_op_kwargs,
                 nonlin=self.nonlin, nonlin_kwargs=self.nonlin_kwargs,basic_block=ConvDropoutNormNonlin,dropout=True)  #128
        self.down_tr256 = DownTransition(input_features=8*channels,num_conv=2,first_stride=1,conv_op=self.conv_op, conv_kwargs=self.conv_kwargs,
                 norm_op=self.norm_op, norm_op_kwargs=self.norm_op_kwargs,
                 dropout_op=self.dropout_op, dropout_op_kwargs=self.dropout_op_kwargs,
                 nonlin=self.nonlin, nonlin_kwargs=self.nonlin_kwargs,basic_block=ConvDropoutNormNonlin,dropout=False)  #256

        self.recons1 = ReconsTransition(input_features=2*channels,output_features=channels,conv_op=self.conv_op, conv_kwargs=self.conv_kwargs,
                 norm_op=self.norm_op, norm_op_kwargs=self.norm_op_kwargs,
                 dropout_op=self.dropout_op, dropout_op_kwargs=self.dropout_op_kwargs,
                 nonlin=self.nonlin, nonlin_kwargs=self.nonlin_kwargs )
        self.recons2 = ReconsTransition(input_features=4*channels,output_features=2*channels,conv_op=self.conv_op, conv_kwargs=self.conv_kwargs,
                 norm_op=self.norm_op, norm_op_kwargs=self.norm_op_kwargs,
                 dropout_op=self.dropout_op, dropout_op_kwargs=self.dropout_op_kwargs,
                 nonlin=self.nonlin, nonlin_kwargs=self.nonlin_kwargs )


        self.att_block1 = Deep_Attention_block(F_up=16 * channels, F_down=8 * channels, F_out=8 * channels,conv_op=self.conv_op, conv_kwargs=self.conv_kwargs,
                 norm_op=self.norm_op, norm_op_kwargs=self.norm_op_kwargs,
                 dropout_op=self.dropout_op, dropout_op_kwargs=self.dropout_op_kwargs,
                 nonlin=self.nonlin, nonlin_kwargs=self.nonlin_kwargs )
        self.att_block2 = Deep_Attention_block(F_up=16 * channels, F_down=4 * channels, F_out=4 * channels,conv_op=self.conv_op, conv_kwargs=self.conv_kwargs,
                 norm_op=self.norm_op, norm_op_kwargs=self.norm_op_kwargs,
                 dropout_op=self.dropout_op, dropout_op_kwargs=self.dropout_op_kwargs,
                 nonlin=self.nonlin, nonlin_kwargs=self.nonlin_kwargs )
        self.att_block3 = Deep_Attention_block(F_up=8* channels, F_down=2 * channels, F_out=2 * channels,conv_op=self.conv_op, conv_kwargs=self.conv_kwargs,
                 norm_op=self.norm_op, norm_op_kwargs=self.norm_op_kwargs,
                 dropout_op=self.dropout_op, dropout_op_kwargs=self.dropout_op_kwargs,
                 nonlin=self.nonlin, nonlin_kwargs=self.nonlin_kwargs )
        #self.att_block4 = Deep_Attention_block(4 * channels, 1 * channels, 1 * channels)

        self.up_tr256 = UpTransition(input_features=16*channels, output_features=16*channels,num_conv=2, first_stride=1,conv_op=self.conv_op, conv_kwargs=self.conv_kwargs,
                 norm_op=self.norm_op, norm_op_kwargs=self.norm_op_kwargs,
                 dropout_op=self.dropout_op, dropout_op_kwargs=self.dropout_op_kwargs,
                 nonlin=self.nonlin, nonlin_kwargs=self.nonlin_kwargs, basic_block=ConvDropoutNormNonlin,dropout=True)
        self.up_tr128 = UpTransition(input_features=16*channels, output_features=8*channels,num_conv=2, first_stride=2,conv_op=self.conv_op, conv_kwargs=self.conv_kwargs,
                 norm_op=self.norm_op, norm_op_kwargs=self.norm_op_kwargs,
                 dropout_op=self.dropout_op, dropout_op_kwargs=self.dropout_op_kwargs,
                 nonlin=self.nonlin, nonlin_kwargs=self.nonlin_kwargs, basic_block=ConvDropoutNormNonlin,dropout=True)
        self.up_tr64 =  UpTransition(input_features=8*channels, output_features=4*channels,num_conv=1, first_stride=2,conv_op=self.conv_op, conv_kwargs=self.conv_kwargs,
                 norm_op=self.norm_op, norm_op_kwargs=self.norm_op_kwargs,
                 dropout_op=self.dropout_op, dropout_op_kwargs=self.dropout_op_kwargs,
                 nonlin=self.nonlin, nonlin_kwargs=self.nonlin_kwargs, basic_block=ConvDropoutNormNonlin)
        self.up_tr32 =  UpTransition(input_features=4*channels, output_features=2*channels,num_conv=1, first_stride=2,conv_op=self.conv_op, conv_kwargs=self.conv_kwargs,
                 norm_op=self.norm_op, norm_op_kwargs=self.norm_op_kwargs,
                 dropout_op=self.dropout_op, dropout_op_kwargs=self.dropout_op_kwargs,
                 nonlin=self.nonlin, nonlin_kwargs=self.nonlin_kwargs, basic_block=ConvDropoutNormNonlin,dropout=True)
        self.out_tr = OutputTransition(input_features=2*channels, output_features=num_classes, conv_op=self.conv_op, conv_kwargs=self.conv_kwargs,
                 norm_op=self.norm_op, norm_op_kwargs=self.norm_op_kwargs,
                 dropout_op=self.dropout_op, dropout_op_kwargs=self.dropout_op_kwargs,
                 nonlin=self.nonlin, nonlin_kwargs=self.nonlin_kwargs)

        self.ex_feature1 = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)))
        self.ex_feature2 = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)))
        self.ex_feature3 = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)))
        self.ex_feature4 = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)))
        self.fc1 = nn.Linear(360,128)
        self.fc2 = nn.Sequential(nn.Linear(128,2),nn.LogSoftmax())

    def forward(self, x):
        print('x:', x.shape)
        #down
        seg_outputs = []
        inconv = self.in_tr(x)
        down1 = self.down_tr32(inconv)
        down2 = self.down_tr64(down1)
        down3 = self.down_tr128(down2)
        down4 = self.down_tr256(down3)


        #recons
        recons1_up = F.upsample(down1, size=inconv.size()[2:], mode='trilinear')
        recon1  = self.recons1(recons1_up)

        recons2_up = F.upsample(down2, size=down1.size()[2:], mode='trilinear')
        recon2 = self.recons2(recons2_up)

        #print('inconv:', inconv.shape)
        #print('down1:', down1.shape)
        #print('down2:', down2.shape)
        #print('down3:', down3.shape)
        #print('down4:', down4.shape)
        feature1 = self.ex_feature1(down1)
        feature2 = self.ex_feature2(down2)
        feature3 = self.ex_feature3(down3)
        feature4 = self.ex_feature4(down4)


        #
        up1 = F.upsample(down4, size=down3.size()[2:], mode='trilinear')
        #print('up1:', up1.shape)

        att1 = self.att_block1(down1,down2,down3,down4,up1,down3)
        #print('att1:', att1.shape)
        up1_conv= self.up_tr256(up1, att1)
        #print('up1_conv:', up1_conv.shape)
        #
        up2 = F.upsample(up1_conv, size=down2.size()[2:], mode='trilinear')
       # print('up2:', up2.shape)

        att2 = self.att_block2(down1, down2, down3, down4, up2, down2)
        #print('att2:', att2.shape)

        up2_conv = self.up_tr128(up2, att2)
        #print('up2_conv:', up2_conv.shape)

        #
        up3 = F.upsample(up2_conv, size=down1.size()[2:], mode='trilinear')
        #print('up3:', up3.shape)

        att3 = self.att_block3(down1, down2, down3, down4, up3, down1)
        #print('att3:', att3.shape)

        up3_conv = self.up_tr64(up3, att3)
        #print('up3_conv:', up3_conv.shape)

        up4= F.upsample(up3_conv, size=inconv.size()[2:], mode='trilinear')
        #print('up4:', up4.shape)
        up4_conv = self.up_tr32(up4, inconv)
        #print('up4_conv:', up4_conv.shape)

        seg_out = self.out_tr(up4_conv)
        #print('seg_out:', seg_out.shape)

        features = torch.cat((feature1,feature2,feature3,feature4),1)
        features = features.view(-1)
        #print('features:', features.shape)

        #cls_out = self.fc1(features)
        #cls_out = self.fc2(cls_out)
        #return recon1,inconv,recon2,down1,seg_out,cls_out
        #print("segout.shape",seg_out.shape)
        seg_outputs.append(seg_out)

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]])
        else:
            return seg_outputs[-1]



class StackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):
        '''
        stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers
        :param input_feature_channels:
        :param output_feature_channels:
        :param num_convs:
        :param dilation:
        :param kernel_size:
        :param padding:
        :param dropout:
        :param initial_stride:
        :param conv_op:
        :param norm_op:
        :param dropout_op:
        :param inplace:
        :param neg_slope:
        :param norm_affine:
        :param conv_bias:
        '''
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            *([basic_block(input_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs_first_conv,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)] +
              [basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))

    def forward(self, x):
        return self.blocks(x)


def print_module_training_status(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d) or isinstance(module, nn.Dropout3d) or \
            isinstance(module, nn.Dropout2d) or isinstance(module, nn.Dropout) or isinstance(module, nn.InstanceNorm3d) \
            or isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.InstanceNorm1d) \
            or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d) or isinstance(module,
                                                                                                      nn.BatchNorm1d):
        print(str(module), module.training)


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)


class Generic_myUNet(SegmentationNetwork):
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

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=False, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False):
        """
        basically more flexible than v1, architecture is the same

        Does this look complicated? Nah bro. Functionality > usability

        This does everything you need, including world peace.

        Questions? -> f.isensee@dkfz.de
        """


        """
        
       
        
        """
        super(Generic_myUNet, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling   #True
        self.convolutional_pooling = convolutional_pooling    #True
        self.upscale_logits = upscale_logits  #False
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}  #
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin   #nn.LeakyReLU
        self.nonlin_kwargs = nonlin_kwargs  #{'negative_slope': 1e-2, 'inplace': True}
        self.dropout_op_kwargs = dropout_op_kwargs   #{'p': 0, 'inplace': True}
        self.norm_op_kwargs = norm_op_kwargs      #{'eps': 1e-5, 'affine': True}
        self.weightInitializer = weightInitializer   #InitWeights_He(1e-2)
        self.conv_op = conv_op   # nn.Conv3d

        self.norm_op = norm_op   #   nn.InstanceNorm3d
        self.dropout_op = dropout_op    # nn.Dropout3d

        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
            pool_op = nn.MaxPool2d
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            transpconv = nn.ConvTranspose3d
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

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.seg_outputs = []

        output_features = base_num_features
        input_features = input_channels

        for d in range(num_pool):
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))

            output_features = min(output_features, self.max_num_features)

        # now the bottleneck.
        # determine the first stride
        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        if self.convolutional_upsampling:
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels

        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway
        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            if not self.convolutional_upsampling:
                self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)],
                                          pool_op_kernel_sizes[-(u + 1)], bias=False))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            ))

        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes,
                                            1, 1, 0, 1, 1, seg_output_use_bias))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
            # self.apply(print_module_training_status)

    def forward(self, x):
        skips = []
        seg_outputs = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
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
            num_blocks = (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
            # print(p, map_size, num_feat, tmp)
        return tmp
