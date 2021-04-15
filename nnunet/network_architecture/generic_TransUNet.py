# 3D-UNet model.
# x: 128x128 resolution for 128 frames.      x: 320*320 resolution for 32frames.
"""
@author: liuyiyao
@time: 2020/10/15 下午
"""
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
ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.leaky_relu}

class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]   #12
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)   #64
        self.all_head_size = self.num_attention_heads * self.attention_head_size   #768

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.config = config
        """
        grid_size = 4, 16, 16
        img_size = 64,256,256
        patch_size = 1, 1, 1
        patch_size_real = 16, 16, 16
        n_patches = 1024
        """
        patch_size = (1,1,1)
        n_patches = 608
        in_channels=256
        self.patch_embeddings = Conv3d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        #print("tag1_3", x.size())

        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        #print("tag1_4", x.size())
        x = x.flatten(2)
        #print("tag1_5", x.size())
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        #print("tag1_6", embeddings.size())
        return embeddings


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x= self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x




class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states= layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        config = ml_collections.ConfigDict()
        config.patches = ml_collections.ConfigDict({'size': (16, 16, 16)})
        config.hidden_size = 768
        config.transformer = ml_collections.ConfigDict()
        config.transformer.mlp_dim = 3072
        config.transformer.num_heads = 12
        config.transformer.num_layers = 12
        config.transformer.attention_dropout_rate = 0.0
        config.transformer.dropout_rate = 0.1
        config.representation_size = None
        config.resnet_pretrained_path = None
        config.patch_size = 16
        config.n_classes = 1
        config.patches.grid = (4, 16, 16)
        config.classifier = 'seg'
        config.decoder_channels = (256, 128, 64, 16)
        config.skip_channels = [512, 256, 64, 16]
        config.n_skip = 3
        config.activation = 'softmax'
        vis = False
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config, vis)
        self.decoder = DecoderCup(config)
    def forward(self, input_ids,h,w,l):
        embedding_output= self.embeddings(input_ids)
        #print("/////////////////////////////")
        #print(embedding_output.size())
        encoded = self.encoder(embedding_output)  # (B, n_patch, hidden)

        #print(encoded.size())
        decoded = self.decoder(encoded,h,w,l)
        return decoded
class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.LeakyReLU()

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, bn, relu)
class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 256
        self.conv_more = Conv3dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )


    def forward(self, hidden_states,h,w,l):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        #h, w ,l = self.h,self.w,self.l
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w ,l)
        x = self.conv_more(x)
        return x

class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        rate_list = (1, rate, rate)
        self.atrous_convolution = nn.Conv3d(inplanes, planes, kernel_size=3,
                                            stride=1, padding=rate_list, dilation=rate_list)
        self.group_norm = nn.GroupNorm(16, planes)
        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.group_norm(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(16, out_ch),
            nn.PReLU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(16, out_ch),
            nn.PReLU())

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            #nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(16, out_ch),
            nn.PReLU(),
        )

    def forward(self, x):
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(16, F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(16, F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1, 1),
            nn.Sigmoid()
        )

        self.PReLU = nn.PReLU()

    def forward(self, g, x):  #up , down
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.PReLU(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out
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
class AttU_Net(SegmentationNetwork):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """
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
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv3d,
                 norm_op=nn.BatchNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=False, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False):
        super(AttU_Net, self).__init__()
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

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.Maxpool1 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(input_channels, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])
        self.Recons_up_conv1 = up_conv(filters[1], filters[0])
        self.Recons_up_conv2 = up_conv(filters[2],filters[1])
        self.transformer  = Transformer()
        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])
        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])
        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])
        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])
        self.Conv = nn.Conv3d(filters[0], num_classes, kernel_size=1, stride=1, padding=0)
        rates = (1, 6, 12, 18)
        self.aspp1 = ASPP_module(16, 16, rate=rates[0])
        self.aspp2 = ASPP_module(16, 16, rate=rates[1])
        self.aspp3 = ASPP_module(16, 16, rate=rates[2])
        self.aspp4 = ASPP_module(16, 16, rate=rates[3])
        self.aspp_conv = nn.Conv3d(64, 64, 1)
        self.aspp_gn = nn.GroupNorm(32, 64)
        self.Out = nn.Conv3d(64, num_classes, kernel_size=1, stride=1, padding=0)
        #self.active = torch.nn.Sigmoid()


    def forward(self, x):
       # print("x shape",x.shape)
        down1 = self.Conv1(x)
        #print("down1 shape",down1.shape)
        pool1 = self.Maxpool1(down1)
        #print("pool1 shape",pool1.shape)
        down2 = self.Conv2(pool1)
        #print("down2 shape",down2.shape)
        pool2 = self.Maxpool2(down2)
        #print("pool2 shape",pool2.shape)
        down3 = self.Conv3(pool2)
        #print("down3 shape",down3.shape)
        pool3 = self.Maxpool3(down3)
        #print("pool3 shape",pool3.shape)
        down4 = self.Conv4(pool3)
       # print("down4 shape",down4.shape)
        pool4 = self.Maxpool4(down4)
        #print("pool4 shape",pool4.shape)
        down5 = self.Conv5(pool4)
       # print("down5 shape",down5.shape)

        trans = self.transformer(down5,down5.shape[2],down5.shape[3],down5.shape[4])
        #print(x5.shape)
        down5_sample = F.upsample(trans, size=down4.size()[2:], mode='trilinear')
        up5 = self.Up5(down5_sample)
       # print("up5 shape",up5.shape)
        #print(d5.shape)
        att4 = self.Att5(g=up5, x=down4)
       # print("att4 shape",att4.shape)
        up5_cat = torch.cat((att4, up5), dim=1)
        #print("up5_cat shape",up5_cat.shape)
        up5_conv = self.Up_conv5(up5_cat)


        #print("up5_conv shape",up5_conv.shape)
        up5_sample = F.upsample(up5_conv, size=down3.size()[2:], mode='trilinear')
        up4 = self.Up4(up5_sample)
      #  print("up4 shape",up4.shape)
        att3 = self.Att4(g=up4, x=down3)
        #print("att3 shape",att3.shape)
        up4_cat = torch.cat((att3, up4), dim=1)
        #print("up4_cat shape",up4_cat.shape)
        up4_conv = self.Up_conv4(up4_cat)


       # print("up4_conv shape",up4_conv.shape)
        up4_sample = F.upsample(up4_conv, size=down2.size()[2:], mode='trilinear')
        up3 = self.Up3(up4_sample)
      #  print("up3 shape",up3.shape)
        att2 = self.Att3(g=up3, x=down2)
       # print("att2 shape",att2.shape)
        up3_cat = torch.cat((att2, up3), dim=1)
       # print("up3_cat shape",up3_cat.shape)
        up3_conv = self.Up_conv3(up3_cat)

       # print("up3_conv shape",up3_conv.shape)
        up3_sample = F.upsample(up3_conv, size=down1.size()[2:], mode='trilinear')
        up2 = self.Up2(up3_sample)
        #print("up2 shape",up2.shape)
        att1 = self.Att2(g=up2, x=down1)
       # print("att1 shape",att1.shape)
        up2_cat = torch.cat((att1, up2), dim=1)
       # print("up2_cat shape",up2_cat.shape)
        up2_conv = self.Up_conv2(up2_cat)

       # print("up2_conv.shape",up2_conv.shape)
        aspp1 = self.aspp1(up2)
       # print("aspp1 shape",aspp1.shape)
        aspp2 = self.aspp2(up2)
       # print("aspp2 shape",aspp2.shape)
        aspp3 = self.aspp3(up2)
        #print("aspp3 shape",aspp3.shape)
        aspp4 = self.aspp4(up2)
       # print("aspp4 shape",aspp4.shape)
        aspp = torch.cat((aspp1, aspp2, aspp3, aspp4), dim=1)
        #print("aspp shape",aspp.shape)
        aspp = self.aspp_gn(self.aspp_conv(aspp))
        #print("aspp shape",aspp.shape)
        seg_out= self.Out(aspp)
        #print("out2 shape",seg_out.shape)

        seg_outputs = []

      #  out = self.active(out)
        seg_outputs.append(seg_out)

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]])
        else:
            return seg_outputs[-1]

