import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunet.network_architecture.neural_network import SegmentationNetwork


def passthrough(x, **kwargs):
    return x

def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

# normalization between sub-volumes is necessary
# for good performance
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


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, 12, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(12)
        self.relu1 = ELUCons(elu, 12)

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x))
        # split input in to 16 channels
        x16 = torch.cat((x, x, x, x, x, x, x, x,x, x, x,x
                         ), 1)
        out = self.relu1(torch.add(out, x16))
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, stride, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=stride)
        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, stride, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.Conv3d(inChans, outChans // 2, kernel_size=1, stride=1)
        self.bn1 = ContBatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out

class ReconsTransition(nn.Module):
    def __init__(self, inChans, outChans,elu):
        super(ReconsTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, inChans, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(inChans)
        self.conv2 = nn.Conv3d(inChans, outChans, kernel_size=1)
        self.relu1 = ELUCons(elu, 1)
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
    def __init__(self, inChans, elu, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, 2, kernel_size=1)
        self.relu1 = ELUCons(elu, 1)
        if nll:
            self.softmax = F.log_softmax
        else:
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
    def __init__(self, F_up, F_down, F_out):
        super(Deep_Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_up, F_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(12, F_out)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_down, F_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(12, F_out),
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(360, 3*F_out, kernel_size=1), nn.GroupNorm( 3*F_out, 3*F_out), nn.PReLU(),
            nn.Conv3d(3*F_out, 2*F_out, kernel_size=3, padding=1), nn.GroupNorm( 2*F_out, 2*F_out), nn.PReLU(),
            nn.Conv3d(2*F_out, F_out, kernel_size=3, padding=1), nn.GroupNorm(F_out, F_out), nn.PReLU()
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_out, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1, 1),
            nn.Sigmoid()
        )
        self.out_att = nn.Sequential(
            nn.Conv3d(2*F_out, F_out, kernel_size=1), nn.GroupNorm(F_out, F_out), nn.PReLU(),
            nn.Conv3d(F_out, F_out, kernel_size=3, padding=1), nn.GroupNorm(F_out, F_out), nn.PReLU(),
            nn.Conv3d(F_out, F_out, kernel_size=3, padding=1), nn.GroupNorm(F_out, F_out), nn.PReLU())
        self.PReLU = nn.PReLU()

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

class VNet(SegmentationNetwork):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=True, nll=False):
        super(VNet, self).__init__()
        channels = 16
        self.in_tr = InputTransition(channels, elu) #16
        self.down_tr32 = DownTransition(channels, 1, elu, 2) #32
        self.down_tr64 = DownTransition(2*channels, 2, elu, 2) #64 
        self.down_tr128 = DownTransition(4*channels, 3, elu, 2, dropout=True) #128
        self.down_tr256 = DownTransition(8*channels, 2, elu, 1, dropout=True) #256

        #self.recons1 = UpTransition(2*channels, channels, 1, elu, 2)

        self.att_block1 = Deep_Attention_block(16 * channels, 8 * channels, 8 * channels)
        self.att_block2 = Deep_Attention_block(16* channels, 4 * channels, 4 * channels)
        self.att_block3 = Deep_Attention_block(8 * channels, 2 * channels, 2 * channels)
        #self.att_block4 = Deep_Attention_block(4 * channels, 1 * channels, 1 * channels)

        self.up_tr256 = UpTransition(16*channels, 16*channels, 2, elu, 1, dropout=True)
        self.up_tr128 = UpTransition(16*channels, 8*channels, 2, elu, 2, dropout=True)
        self.up_tr64 =  UpTransition(8*channels,4*channels, 1, elu, 2)
        self.up_tr32 =  UpTransition(4*channels, 2*channels, 1, elu, 2)
        self.out_tr = OutputTransition(2*channels, elu, nll)

        self.ex_feature1 = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)))
        self.ex_feature2 = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)))
        self.ex_feature3 = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)))
        self.ex_feature4 = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)))
        self.fc1 = nn.Linear(360,256)
        self.fc2 = nn.Sequential(nn.Linear(256,2),nn.LogSoftmax())

    def forward(self, x):
        print('x:', x.shape)
        #down
        inconv = self.in_tr(x)
        down1 = self.down_tr32(inconv)
        down2 = self.down_tr64(down1)
        down3 = self.down_tr128(down2)
        down4 = self.down_tr256(down3)
        print('inconv:', inconv.shape)
        print('down1:', down1.shape)
        print('down2:', down2.shape)
        print('down3:', down3.shape)
        print('down4:', down4.shape)
        feature1 = self.ex_feature1(down1)
        feature2 = self.ex_feature2(down2)
        feature3 = self.ex_feature3(down3)
        feature4 = self.ex_feature4(down4)


        #
        up1 = F.upsample(down4, size=down3.size()[2:], mode='trilinear')
        print('up1:', up1.shape)

        att1 = self.att_block1(down1,down2,down3,down4,up1,down3)
        print('att1:', att1.shape)
        up1_conv= self.up_tr256(up1, att1)
        print('up1_conv:', up1_conv.shape)
        #
        up2 = F.upsample(up1_conv, size=down2.size()[2:], mode='trilinear')
        print('up2:', up2.shape)

        att2 = self.att_block2(down1, down2, down3, down4, up2, down2)
        print('att2:', att2.shape)

        up2_conv = self.up_tr128(up2, att2)
        print('up2_conv:', up2_conv.shape)

        #
        up3 = F.upsample(up2_conv, size=down1.size()[2:], mode='trilinear')
        print('up3:', up3.shape)

        att3 = self.att_block3(down1, down2, down3, down4, up3, down1)
        print('att3:', att3.shape)

        up3_conv = self.up_tr64(up3, att3)
        print('up3_conv:', up3_conv.shape)

        up4= F.upsample(up3_conv, size=inconv.size()[2:], mode='trilinear')
        print('up4:', up4.shape)
        up4_conv = self.up_tr32(up4, inconv)
        print('up4_conv:', up4_conv.shape)

        seg_out = self.out_tr(up4_conv)
        print('seg_out:', seg_out.shape)

        features = torch.cat((feature1,feature2,feature3,feature4),1)
        features = features.view(-1)
        print('features:', features.shape)

        cls_out = self.fc1(features)
        cls_out = self.fc2(cls_out)
        return seg_out,cls_out

class VNet_recons(SegmentationNetwork):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=True, nll=False):
        super(VNet_recons, self).__init__()
        channels = 12
        self.in_tr = InputTransition(channels, elu) #16
        self.down_tr32 = DownTransition(channels, 1, elu, 2) #32
        self.down_tr64 = DownTransition(2*channels, 2, elu, 2) #64
        self.down_tr128 = DownTransition(4*channels, 3, elu, 2, dropout=True) #128
        self.down_tr256 = DownTransition(8*channels, 2, elu, 1, dropout=True) #256

        self.recons1 = ReconsTransition(2*channels, channels, elu)
        self.recons2 = ReconsTransition(4*channels, 2*channels, elu)


        self.att_block1 = Deep_Attention_block(16 * channels, 8 * channels, 8 * channels)
        self.att_block2 = Deep_Attention_block(16* channels, 4 * channels, 4 * channels)
        self.att_block3 = Deep_Attention_block(8 * channels, 2 * channels, 2 * channels)
        #self.att_block4 = Deep_Attention_block(4 * channels, 1 * channels, 1 * channels)

        self.up_tr256 = UpTransition(16*channels, 16*channels, 2, elu, 1, dropout=True)
        self.up_tr128 = UpTransition(16*channels, 8*channels, 2, elu, 2, dropout=True)
        self.up_tr64 =  UpTransition(8*channels,4*channels, 1, elu, 2)
        self.up_tr32 =  UpTransition(4*channels, 2*channels, 1, elu, 2)
        self.out_tr = OutputTransition(2*channels, elu, nll)

        self.ex_feature1 = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)))
        self.ex_feature2 = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)))
        self.ex_feature3 = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)))
        self.ex_feature4 = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)))
        self.fc1 = nn.Linear(360,128)
        self.fc2 = nn.Sequential(nn.Linear(128,2),nn.LogSoftmax())

    def forward(self, x):
        #print('x:', x.shape)
        #down
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
        return seg_out
class VNet_recons_noam(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=True, nll=False):
        super(VNet_recons_noam, self).__init__()
        channels = 12
        self.in_tr = InputTransition(channels, elu) #16
        self.down_tr32 = DownTransition(channels, 1, elu, 2) #32
        self.down_tr64 = DownTransition(2*channels, 2, elu, 2) #64
        self.down_tr128 = DownTransition(4*channels, 3, elu, 2, dropout=True) #128
        self.down_tr256 = DownTransition(8*channels, 2, elu, 1, dropout=True) #256

        self.recons1 = ReconsTransition(2*channels, channels, elu)
        self.recons2 = ReconsTransition(4*channels, 2*channels, elu)


        self.att_block1 = Deep_Attention_block(16 * channels, 8 * channels, 8 * channels)
        self.att_block2 = Deep_Attention_block(16* channels, 4 * channels, 4 * channels)
        self.att_block3 = Deep_Attention_block(8 * channels, 2 * channels, 2 * channels)
        #self.att_block4 = Deep_Attention_block(4 * channels, 1 * channels, 1 * channels)

        self.up_tr256 = UpTransition(16*channels, 16*channels, 2, elu, 1, dropout=True)
        self.up_tr128 = UpTransition(16*channels, 8*channels, 2, elu, 2, dropout=True)
        self.up_tr64 =  UpTransition(8*channels,4*channels, 1, elu, 2)
        self.up_tr32 =  UpTransition(4*channels, 2*channels, 1, elu, 2)
        self.out_tr = OutputTransition(2*channels, elu, nll)

        self.ex_feature1 = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)))
        self.ex_feature2 = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)))
        self.ex_feature3 = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)))
        self.ex_feature4 = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)))
        self.fc1 = nn.Linear(360,128)
        self.fc2 = nn.Sequential(nn.Linear(128,2),nn.LogSoftmax())

    def forward(self, x):
        print('x:', x.shape)
        #down
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

        print('inconv:', inconv.shape)
        print('down1:', down1.shape)
        print('down2:', down2.shape)
        print('down3:', down3.shape)
        print('down4:', down4.shape)
        feature1 = self.ex_feature1(down1)
        feature2 = self.ex_feature2(down2)
        feature3 = self.ex_feature3(down3)
        feature4 = self.ex_feature4(down4)


        #
        up1 = F.upsample(down4, size=down3.size()[2:], mode='trilinear')
        print('up1:', up1.shape)


        up1_conv= self.up_tr256(up1, down3)
        print('up1_conv:', up1_conv.shape)
        #
        up2 = F.upsample(up1_conv, size=down2.size()[2:], mode='trilinear')
        print('up2:', up2.shape)



        up2_conv = self.up_tr128(up2, down2)
        print('up2_conv:', up2_conv.shape)

        #
        up3 = F.upsample(up2_conv, size=down1.size()[2:], mode='trilinear')
        print('up3:', up3.shape)


        up3_conv = self.up_tr64(up3, down1)
        print('up3_conv:', up3_conv.shape)

        up4= F.upsample(up3_conv, size=inconv.size()[2:], mode='trilinear')
        print('up4:', up4.shape)
        up4_conv = self.up_tr32(up4, inconv)
        print('up4_conv:', up4_conv.shape)

        seg_out = self.out_tr(up4_conv)
        print('seg_out:', seg_out.shape)

        features = torch.cat((feature1,feature2,feature3,feature4),1)
        features = features.view(-1)
        print('features:', features.shape)

        cls_out = self.fc1(features)
        cls_out = self.fc2(cls_out)
        #return recon1,inconv,recon2,down1,seg_out,cls_out
        return seg_out