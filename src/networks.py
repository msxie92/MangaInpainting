import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn
from PIL import Image
from torchvision import transforms
from torchvision import utils as vutils
from torch.nn.modules.normalization import LayerNorm

from utils.tools import extract_image_patches, flow_to_image, \
    reduce_mean, reduce_sum, default_loader, same_padding


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

class SemanticInpaintGenerator(BaseNetwork):
    def __init__(self, in_channels=3, out_channels=4, residual_blocks=6, init_weights=True):
        super(SemanticInpaintGenerator, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, padding=0)),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            LayerNormWrapper(64),
            nn.ReLU(True))

        self.encoder2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            LayerNormWrapper(128),
            nn.ReLU(True))

        self.encoder3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            LayerNormWrapper(256),
            nn.ReLU(True))

        self.encoder4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            LayerNormWrapper(256),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2, use_instance=True)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder_scr4 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            # LayerNormWrapper(256),
            nn.ReLU(True))

        self.decoder_scr3 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            # LayerNormWrapper(128),
            nn.ReLU(True))

        self.decoder_scr2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            # LayerNormWrapper(64),
            nn.ReLU(True))

        self.decoder_scr1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=7, padding=0),
        )

        self.decoder_l4 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            LayerNormWrapper(256),
            nn.ReLU(True))

        self.decoder_l3 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            LayerNormWrapper(128),
            nn.ReLU(True))

        self.decoder_l2 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            LayerNormWrapper(64),
            nn.ReLU(True))

        self.decoder_l1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, xin):
        x1 = self.encoder1(xin)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x = self.middle(x4)

        scr = self.decoder_scr4(x)
        scr = self.decoder_scr3(scr)
        scr = self.decoder_scr2(scr)
        scr = self.decoder_scr1(scr)
        scr = torch.clamp(scr, -1., 1.)

        line = self.decoder_l4(x+x4)
        line = self.decoder_l3(line+x3)
        line = self.decoder_l2(line+x2)
        line = self.decoder_l1(line+x1)
        line = torch.clamp(line, -1., 1.)

        return scr, line


class MangaInpaintGenerator(BaseNetwork):
    def __init__(self, input_dim=6, cnum=32, output_dim=1, init_weights=True):#, use_cuda=True, device_ids=None):
        super(MangaInpaintGenerator, self).__init__()
        # self.use_cuda = use_cuda
        # self.device_ids = device_ids
        # input_dim=6
        # output_dim=1

        # 3 x 256 x 256
        self.conv1 = gen_conv(input_dim + 2, cnum, 5, 1, 2)
        self.conv2_downsample = gen_conv(cnum, cnum, 3, 2, 1)
        # cnum*2 x 128 x 128
        self.conv3 = gen_conv(cnum, cnum*2, 3, 1, 1)
        self.conv4_downsample = gen_conv(cnum*2, cnum*2, 3, 2, 1)
        # cnum*4 x 64 x 64
        self.conv5 = gen_conv(cnum*2, cnum*4, 3, 1, 1)
        self.conv6_downsample = gen_conv(cnum*4, cnum*4, 3, 1, 1)

        self.conv7_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 2, rate=2)
        self.conv8_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 4, rate=4)
        self.conv9_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 8, rate=8)
        self.conv10_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 16, rate=16)

        # attention branch
        # 3 x 256 x 256
        self.pmconv1 = gen_conv(input_dim + 2, cnum, 5, 1, 2)
        self.pmconv2_downsample = gen_conv(cnum, cnum, 3, 2, 1)
        # cnum*2 x 128 x 128
        self.pmconv3 = gen_conv(cnum, cnum*2, 3, 1, 1)
        self.pmconv4_downsample = gen_conv(cnum*2, cnum*4, 3, 2, 1)
        # cnum*4 x 64 x 64
        self.pmconv5 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.pmconv6 = gen_conv(cnum*4, cnum*4, 3, 1, 1, activation='relu')
        self.contextul_attention = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10,
                                                       fuse=True)#, use_cuda=self.use_cuda, device_ids=self.device_ids)
        self.pmconv9 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.pmconv10 = gen_conv(cnum*4, cnum*4, 3, 1, 1)

        self.allconv11 = gen_conv(cnum*8, cnum*4, 3, 1, 1)
        self.allconv12 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.allconv13 = gen_deconv(cnum*4, cnum*2, 3, 2, 1)
        self.allconv14 = gen_conv(cnum*2, cnum*2, 3, 1, 1)
        self.allconv15 = gen_deconv(cnum*2, cnum, 3, 2, 1)
        self.allconv16 = gen_conv(cnum, cnum//2, 3, 1, 1)
        self.allconv17 = gen_conv(cnum//2, output_dim, 5, 1, 2, activation='none')
        # self.allconv17 = gen_conv(cnum//2, output_dim, 3, 1, 1, activation='none')

        if init_weights:
            self.init_weights()

    def forward(self, xin, mask):
        # For indicating the boundaries of images
        # ones = torch.ones(xin.size(0), 1, xin.size(2), xin.size(3))
        # if self.use_cuda:
        #     ones = ones.cuda()
        #     mask = mask.cuda()
        ones = torch.ones_like(mask)
        # conv branch
        xnow = torch.cat([xin, ones, mask], dim=1)
        x = self.conv1(xnow)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6_downsample(x)

        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x_hallu = x
        # attention branch
        x = self.pmconv1(xnow)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)
        x = self.pmconv6(x)
        x, offset_flow = self.contextul_attention(x, x, mask)
        x = self.pmconv9(x)
        x = self.pmconv10(x)
        pm = x
        x = torch.cat([x_hallu, pm], dim=1)
        # merge two branches
        x = self.allconv11(x)
        x = self.allconv12(x)
        # x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv13(x)
        x = self.allconv14(x)
        # x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv15(x)
        x = self.allconv16(x)
        x = self.allconv17(x)
        # x_stage2 = torch.tanh(x)
        x_stage2 = torch.clamp(x, -1., 1.)

        return x_stage2


class ContextualAttention(nn.Module):
    def __init__(self, ksize=3, stride=1, rate=1, fuse_k=3, softmax_scale=10,
                 fuse=False):#, use_cuda=False, device_ids=None):
        super(ContextualAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        # self.use_cuda = use_cuda
        # self.device_ids = device_ids

    def forward(self, f, b, mask=None):
        """ Contextual attention layer implementation.
        Contextual attention is first introduced in publication:
            Generative Image Inpainting with Contextual Attention, Yu et al.
        Args:
            f: Input feature to match (foreground).
            b: Input feature for match (background).
            mask: Input mask for b, indicating patches not available.
            ksize: Kernel size for contextual attention.
            stride: Stride for extracting patches from b.
            rate: Dilation for matching.
            softmax_scale: Scaled softmax for attention.
        Returns:
            torch.tensor: output
        """
        # get shapes
        raw_int_fs = list(f.size())   # b*c*h*w
        raw_int_bs = list(b.size())   # b*c*h*w

        # extract patches from background with stride and rate
        kernel = 2 * self.rate
        # raw_w is extracted for reconstruction
        raw_w = extract_image_patches(b, ksizes=[kernel, kernel],
                                      strides=[self.rate*self.stride,
                                               self.rate*self.stride],
                                      rates=[1, 1],
                                      padding='same') # [N, C*k*k, L]
        # raw_shape: [N, C, k, k, L]
        raw_w = raw_w.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)    # raw_shape: [N, L, C, k, k]
        raw_w_groups = torch.split(raw_w, 1, dim=0)

        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        f = F.interpolate(f, scale_factor=1./self.rate, mode='nearest')
        b = F.interpolate(b, scale_factor=1./self.rate, mode='nearest')
        int_fs = list(f.size())     # b*c*h*w
        int_bs = list(b.size())
        f_groups = torch.split(f, 1, dim=0)  # split tensors along the batch dimension
        # w shape: [N, C*k*k, L]
        w = extract_image_patches(b, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        # w shape: [N, C, k, k, L]
        w = w.view(int_bs[0], int_bs[1], self.ksize, self.ksize, -1)
        w = w.permute(0, 4, 1, 2, 3)    # w shape: [N, L, C, k, k]
        w_groups = torch.split(w, 1, dim=0)

        # process mask
        if mask is None:
            mask = torch.zeros([int_bs[0], 1, int_bs[2], int_bs[3]]).to(f.device)
            # if self.use_cuda:
            #     mask = mask.cuda()
        else:
            mask = F.interpolate(mask, scale_factor=1./(4*self.rate), mode='nearest')
        int_ms = list(mask.size())
        # m shape: [N, C*k*k, L]
        m = extract_image_patches(mask, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        # m shape: [N, C, k, k, L]
        m = m.view(int_ms[0], int_ms[1], self.ksize, self.ksize, -1)
        m = m.permute(0, 4, 1, 2, 3)    # m shape: [N, L, C, k, k]
        m = m[0]    # m shape: [L, C, k, k]
        # mm shape: [L, 1, 1, 1]
        mm = (reduce_mean(m, axis=[1, 2, 3], keepdim=True)==0.).to(torch.float32)
        mm = mm.permute(1, 0, 2, 3) # mm shape: [1, L, 1, 1]

        y = []
        offsets = []
        k = self.fuse_k
        scale = self.softmax_scale    # to fit the PyTorch tensor image value range
        fuse_weight = torch.eye(k).view(1, 1, k, k).to(f.device)  # 1*1*k*k
        # if self.use_cuda:
        #     fuse_weight = fuse_weight.cuda()

        for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
            '''
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
            wi : separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=3, KW=3)
            raw_wi : separated tensor along batch dimension of back; (B=1, I=32*32, O=128, KH=4, KW=4)
            '''
            # conv for compare
            escape_NaN = torch.FloatTensor([1e-4]).to(f.device)
            # if self.use_cuda:
            #     escape_NaN = escape_NaN.cuda()
            wi = wi[0]  # [L, C, k, k]
            max_wi = torch.sqrt(reduce_sum(torch.pow(wi, 2) + escape_NaN, axis=[1, 2, 3], keepdim=True))
            wi_normed = wi / max_wi
            # xi shape: [1, C, H, W], yi shape: [1, L, H, W]
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)   # [1, L, H, W]
            # conv implementation for fuse scores to encourage large patches
            if self.fuse:
                # make all of depth to spatial resolution
                yi = yi.view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])  # (B=1, I=1, H=32*32, W=32*32)
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)  # (B=1, C=1, H=32*32, W=32*32)
                yi = yi.contiguous().view(1, int_bs[2], int_bs[3], int_fs[2], int_fs[3])  # (B=1, 32, 32, 32, 32)
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = yi.contiguous().view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)
                yi = yi.contiguous().view(1, int_bs[3], int_bs[2], int_fs[3], int_fs[2])
                yi = yi.permute(0, 2, 1, 4, 3).contiguous()
            yi = yi.view(1, int_bs[2] * int_bs[3], int_fs[2], int_fs[3])  # (B=1, C=32*32, H=32, W=32)
            # softmax to match
            yi = yi * mm
            yi = F.softmax(yi*scale, dim=1)
            yi = yi * mm  # [1, L, H, W]

            offset = torch.argmax(yi, dim=1, keepdim=True)  # 1*1*H*W

            if int_bs != int_fs:
                # Normalize the offset value to match foreground dimension
                times = float(int_fs[2] * int_fs[3]) / float(int_bs[2] * int_bs[3])
                offset = ((offset + 1).float() * times - 1).to(torch.int64)
            offset = torch.cat([offset//int_fs[3], offset%int_fs[3]], dim=1)  # 1*2*H*W

            # deconv for patch pasting
            wi_center = raw_wi[0]
            # yi = F.pad(yi, [0, 1, 0, 1])    # here may need conv_transpose same padding
            yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1) / 4.  # (B=1, C=128, H=64, W=64)
            y.append(yi)
            offsets.append(offset)

        y = torch.cat(y, dim=0)  # back to the mini-batch
        y.contiguous().view(raw_int_fs)

        offsets = torch.cat(offsets, dim=0)
        offsets = offsets.view(int_fs[0], 2, *int_fs[2:])

        # case1: visualize optical flow: minus current position
        h_add = torch.arange(int_fs[2]).view([1, 1, int_fs[2], 1]).expand(int_fs[0], -1, -1, int_fs[3])
        w_add = torch.arange(int_fs[3]).view([1, 1, 1, int_fs[3]]).expand(int_fs[0], -1, int_fs[2], -1)
        ref_coordinate = torch.cat([h_add, w_add], dim=1).to(f.device)
        # if self.use_cuda:
        #     ref_coordinate = ref_coordinate.cuda()

        ref_offsets = offsets - ref_coordinate
        # flow = pt_flow_to_image(offsets)

        flow = torch.from_numpy(flow_to_image(ref_offsets.permute(0, 2, 3, 1).cpu().data.numpy())) / 255.
        flow = flow.permute(0, 3, 1, 2).to(f.device)
        # if self.use_cuda:
        #     flow = flow.cuda()
        # case2: visualize which pixels are attended
        # flow = torch.from_numpy(highlight_flow((ref_offsets * mask.long()).cpu().data.numpy()))

        if self.rate != 1:
            flow = F.interpolate(flow, scale_factor=self.rate*4, mode='nearest')

        return y, offsets
        return y, flow


def gen_conv(input_dim, output_dim, kernel_size=3, stride=1, padding=0, rate=1,
             activation='elu'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       padding=padding, dilation=rate,
                       activation=activation)

def gen_deconv(input_dim, output_dim, kernel_size=3, stride=1, padding=0, rate=1,
             activation='elu'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation, transpose=True)

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0,
                 conv_padding=0, dilation=1, weight_norm='none', norm='none',
                 activation='relu', pad_type='replicate', transpose=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNormWrapper(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if weight_norm == 'sn':
            self.weight_norm = spectral_norm_fn
        elif weight_norm == 'wn':
            self.weight_norm = weight_norm_fn
        elif weight_norm == 'none':
            self.weight_norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(weight_norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if transpose:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim,
                                           kernel_size, stride,
                                           padding=conv_padding,
                                           output_padding=conv_padding,
                                           dilation=dilation,
                                           bias=self.use_bias)
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                                  padding=conv_padding, dilation=dilation,
                                  bias=self.use_bias, padding_mode=pad_type)

        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)

    def forward(self, x):
        if self.pad:
            x = self.conv(self.pad(x))
        else:
            x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False, use_instance=False):
        super(ResnetBlock, self).__init__()
        if use_instance:
            self.conv_block = nn.Sequential(
                nn.ReflectionPad2d(dilation),
                spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
                # nn.InstanceNorm2d(dim, track_running_stats=False),
                LayerNormWrapper(dim),
                nn.ReLU(True),

                nn.ReflectionPad2d(1),
                spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
                # nn.InstanceNorm2d(dim, track_running_stats=False),
                LayerNormWrapper(dim),
            )
        else:
            self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(dim, track_running_stats=False),
            # LayerNormWrapper(dim),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(dim, track_running_stats=False),
            # LayerNormWrapper(dim),
        )


    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


class LayerNormWrapper(nn.Module):
    def __init__(self, num_features):
        super(LayerNormWrapper, self).__init__()
        self.num_features = int(num_features)

    def forward(self, x):
        #y = x.to(torch.float32)
        # x = nn.LayerNorm([self.num_features, x.size()[2], x.size()[3]]).cuda()(x)#.to(x.dtype)
        x = nn.LayerNorm([self.num_features, x.size()[2], x.size()[3]], elementwise_affine=False).cuda()(x)
        return x


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

