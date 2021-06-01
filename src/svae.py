import os
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm

class ScreenVAE(nn.Module):
    def __init__(self,inc=1,outc=4, outplanes=64, downs=5, blocks=2,load_ext=True, 
        save_dir='checkpoints/ScreenVAE', init_type="normal", init_gain=0.02, gpu_ids=[]):
        super(ScreenVAE, self).__init__()
        self.inc = inc
        self.outc = outc
        self.save_dir = save_dir
        self.model_names=['enc','dec']
        self.enc=define_C(inc+1, outc*2, 24, netC='resnet_6blocks', 
                                      norm='layer', nl='lrelu', use_dropout=True, 
                                      gpu_ids=gpu_ids, upsample='bilinear')
        self.dec=define_G(outc, inc, 48, netG='unet_128_G', 
                                      norm='layer', nl='lrelu', use_dropout=True, 
                                      gpu_ids=gpu_ids, where_add='input', upsample='bilinear', use_noise=True)
        self.load_networks('latest')
        for param in self.parameters():
            param.requires_grad = False

    def load_networks(self, epoch):

        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(
                    load_path, map_location=lambda storage, loc: storage.cuda())
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                net.load_state_dict(state_dict)
                del state_dict

    def load_gaborext(self, gpu_ids=[]):
        self.gaborext = GaborWavelet()
        self.gaborext.eval()
        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.gaborext.to(gpu_ids[0])

    def npad(self, im, pad=128):
        h,w = im.shape[-2:]
        hp = (h+pad-1) //pad*pad
        wp = (w+pad-1) //pad*pad
        return F.pad(im, (0, wp-w, 0, hp-h), mode='constant',value=1)

    def forward(self, x, line=None, screen=False, rep=False):
        if line is None:
            line = torch.ones_like(x)
        else:
            line = torch.sign(line)
            x = torch.clamp(x + (1-line),-1,1)
        if not screen:
            input = torch.cat([x, line], 1)
            # input = F.pad(input, (32, 32, 32, 32), mode='constant',value=1)
            inter = self.enc(input)
            scr, logvar = torch.split(inter, (self.outc, self.outc), dim=1)#[:,:,32:-32,32:-32]
            # scr = scr*torch.sign(line+1)
            if rep:
                return scr
            recons = self.dec(scr)
            return recons, scr#, logvar
        else:
            h,w = x.shape[-2:]
            x = self.npad(x)
            recons = self.dec(x)[:,:,:h,:w]
            recons = (recons+1)*(line+1)/2-1
            return torch.clamp(recons,-1,1)

###############################################################################
# Helper functions
###############################################################################

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                #init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                init.kaiming_normal_(m.weight.data, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], init=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
    if init:
        init_weights(net, init_type, init_gain=init_gain)
    return net

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'layer':
        norm_layer = functools.partial(LayerNormWarpper)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class LayerNormWarpper(nn.Module):
    def __init__(self, num_features):
        super(LayerNormWarpper, self).__init__()
        self.num_features = int(num_features)

    def forward(self, x):
        x = nn.LayerNorm([self.num_features, x.size()[2], x.size()[3]], elementwise_affine=False).cuda()(x)
        return x

def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    elif layer_type == 'selu':
        nl_layer = functools.partial(nn.SELU, inplace=True)
    elif layer_type == 'prelu':
        nl_layer = functools.partial(nn.PReLU)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer


class ApplyNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.weight = nn.Parameter(torch.randn(channels), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(channels), requires_grad=True)

    def forward(self, x, noise):
        W,_ = torch.split(self.weight.view(1, -1, 1, 1), self.channels // 2, dim=1)
        B,_ = torch.split(self.bias.view(1, -1, 1, 1), self.channels // 2, dim=1)
        Z = torch.zeros_like(W)
        w = torch.cat([W,Z], dim=1).to(x.device)
        b = torch.cat([B,Z], dim=1).to(x.device)
        adds = w * torch.randn_like(x) + b
        return x + adds.type_as(x)

def define_G(input_nc, output_nc, ngf, netG='unet_128', norm='layer', nl='lrelu', use_noise=False, level=0,
             use_dropout=False, init_type='xavier', init_gain=0.02, gpu_ids=[], where_add='input', upsample='bilinear'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    nl_layer = get_non_linearity(layer_type=nl)

    if netG == 'unet_128_G' and where_add == 'input':
        net = G_Unet_add_input_G(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, nl_layer=nl_layer, use_noise=use_noise,
                               use_dropout=use_dropout, upsample=upsample, device=gpu_ids)
    elif netG == 'unet_256_G' and where_add == 'input':
        net = G_Unet_add_input_G(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, nl_layer=nl_layer, use_noise=use_noise,
                               use_dropout=use_dropout, upsample=upsample, device=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_C(input_nc, output_nc, ngf, netC='unet_128', norm='instance', nl='relu',
             use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], upsample='basic'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    nl_layer = get_non_linearity(layer_type=nl)

    if netC == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netC == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net)

    return init_net(net, init_type, init_gain, gpu_ids)

class G_Unet_add_input_G(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, 
                 norm_layer=None, nl_layer=None, use_dropout=False, use_noise=False,
                 upsample='basic', device=0):
        super(G_Unet_add_input_G, self).__init__()
        max_nchn = 8
        # construct unet structure
        #print(num_downs)
        unet_block = UnetBlock(ngf * max_nchn, ngf * max_nchn, ngf * max_nchn, noise=False,
                               innermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        for i in range(num_downs - 5):
            unet_block = UnetBlock(ngf * max_nchn, ngf * max_nchn, ngf * max_nchn, unet_block, noise=False,
                                   norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
        unet_block = UnetBlock(ngf * 4, ngf * 4, ngf * max_nchn, unet_block, use_noise,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample='basic')
        unet_block = UnetBlock(ngf * 2, ngf * 2, ngf * 4, unet_block, use_noise,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample='basic')
        unet_block = UnetBlock(ngf, ngf, ngf * 2, unet_block, use_noise,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample='basic')
        unet_block = UnetBlock(input_nc, output_nc, ngf, unet_block, noise=False,
                               outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample='basic')

        self.model = unet_block

    def forward(self, x):
        # return torch.tanh(self.model(x))
        return self.model(x)


def upsampleLayer(inplanes, outplanes, kw=1, upsample='basic', padding_type='replicate'):
    # padding_type = 'zero'
    if upsample == 'basic':
        upconv = [nn.ConvTranspose2d(inplanes, outplanes, kernel_size=4, stride=2, padding=1)]#, padding_mode='replicate'
    elif upsample == 'bilinear' or upsample == 'nearest' or upsample == 'linear':
        upconv = [nn.Upsample(scale_factor=2, mode=upsample, align_corners=True),
                  #nn.ReplicationPad2d(1),
                  nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)]
        # p = kw//2
        # upconv = [nn.Upsample(scale_factor=2, mode=upsample, align_corners=True),
        #           nn.Conv2d(inplanes, outplanes, kernel_size=kw, stride=1, padding=p, padding_mode='replicate')]
    else:
        raise NotImplementedError(
            'upsample layer [%s] not implemented' % upsample)
    return upconv

class UnetBlock(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc,
                 submodule=None, noise=None, outermost=False, innermost=False, 
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic', padding_type='replicate'):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        p = 0
        downconv = []
        if padding_type == 'reflect':
            downconv += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        downconv += [nn.Conv2d(input_nc, inner_nc,
                               kernel_size=3, stride=2, padding=p)]
        # downsample is different from upsample
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc) if norm_layer is not None else None
        uprelu = nl_layer()
        uprelu2 = nl_layer()
        uppad = nn.ReplicationPad2d(1)
        upnorm = norm_layer(outer_nc) if norm_layer is not None else None
        upnorm2 = norm_layer(outer_nc) if norm_layer is not None else None
        self.noiseblock = ApplyNoise(outer_nc)
        self.noise = noise

        if outermost:
            upconv = upsampleLayer(inner_nc * 2, inner_nc, upsample=upsample, padding_type=padding_type)
            uppad = nn.ReplicationPad2d(3)
            upconv2 = nn.Conv2d(inner_nc, outer_nc, kernel_size=7, padding=0)
            # upconv = upsampleLayer(inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            # upconv2 = nn.Conv2d(outer_nc, outer_nc, kernel_size=3, padding=p)
            down = downconv
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [norm_layer(inner_nc)]
                # up += [norm_layer(outer_nc)]
            up +=[uprelu2, uppad, upconv2] #+ [nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = upsampleLayer(inner_nc, outer_nc, upsample=upsample, padding_type=padding_type)
            upconv2 = nn.Conv2d(outer_nc, outer_nc, kernel_size=3, padding=p)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]
            up += [uprelu2, uppad, upconv2]
            if upnorm2 is not None:
                up += [upnorm2]
            model = down + up
        else:
            upconv = upsampleLayer(inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            upconv2 = nn.Conv2d(outer_nc, outer_nc, kernel_size=3, padding=p)
            down = [downrelu] + downconv
            if downnorm is not None:
                down += [downnorm]
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]
            up += [uprelu2, uppad, upconv2]
            if upnorm2 is not None:
                up += [upnorm2]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            x2 = self.model(x)
            if self.noise:
                x2 = self.noiseblock(x2, self.noise)
            return torch.cat([x2, x], 1)

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, norm_layer=None, use_dropout=True, n_blocks=6, padding_type='replicate'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        model = [nn.ReplicationPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias)]
        if norm_layer is not None:
            model += [norm_layer(ngf)]
        model += [nn.ReLU(True)]

        # n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.ReplicationPad2d(1),nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=0, bias=use_bias)]
            if norm_layer is not None:
                model += [norm_layer(ngf * mult * 2)]
            model += [nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            # model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
            #                              kernel_size=3, stride=2,
            #                              padding=1, output_padding=1,
            #                              bias=use_bias)]
            # if norm_layer is not None:
            #     model += [norm_layer(ngf * mult / 2)]
            # model += [nn.ReLU(True)]
            model += upsampleLayer(ngf * mult, int(ngf * mult / 2), upsample='bilinear', padding_type=padding_type)
            if norm_layer is not None:
                model += [norm_layer(int(ngf * mult / 2))]
            model += [nn.ReLU(True)]
            model +=[nn.ReplicationPad2d(1),
                     nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 2), kernel_size=3, padding=0)]
            if norm_layer is not None:
                model += [norm_layer(ngf * mult / 2)]
            model += [nn.ReLU(True)]
        model += [nn.ReplicationPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        #model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        if norm_layer is not None:
            conv_block += [norm_layer(dim)]
        conv_block += [nn.ReLU(True)]
        # if use_dropout:
        #     conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        if norm_layer is not None:
            conv_block += [norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


