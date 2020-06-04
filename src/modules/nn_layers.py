import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm

from src.utils import args



# ----- Helpers -----

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, unflatten_size):
        super().__init__()
        if isinstance(unflatten_size, tuple):
            self.c = unflatten_size[0]
            self.h = unflatten_size[1]
            self.w = unflatten_size[2]
        elif isinstance(unflatten_size, int):
            self.c = unflatten_size
            self.h = 1
            self.w = 1

    def forward(self, x):
        return x.view(x.size(0), self.c, self.h, self.w)


# ----- 2D Convolutions -----

# Conv2d init_parameters from: https://github.com/vlievin/biva-pytorch/blob/master/biva/layers/convolution.py
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, weightnorm=True, act=None, drop_prob=0.0):
        super().__init__()
        self.weightnorm = weightnorm
        self.initialized = True

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.act = nn.ELU(inplace=True) if act is not None else Identity()
        self.drop_prob = drop_prob

        if self.weightnorm:
            self.initialized = False
            self.conv = nn.utils.weight_norm(self.conv, dim=0, name="weight")

    def forward(self, input):
        if not self.initialized:
            self.init_parameters(input)
        return F.dropout(self.act(self.conv(input)), p=self.drop_prob, training=True)

    def init_parameters(self, x, init_scale=0.05, eps=1e-8):
        self.initialized = True
        if self.weightnorm:
            # initial values
            self.conv._parameters['weight_v'].data.normal_(mean=0, std=init_scale)
            self.conv._parameters['weight_g'].data.fill_(1.)
            self.conv._parameters['bias'].data.fill_(0.)
            init_scale = .01
            # data dependent init
            x = self.conv(x)
            t = x.view(x.size()[0], x.size()[1], -1)
            t = t.permute(0, 2, 1).contiguous()
            t = t.view(-1, t.size()[-1])
            m_init, v_init = torch.mean(t, 0), torch.var(t, 0)
            scale_init = init_scale / torch.sqrt(v_init + eps)

            self.conv._parameters['weight_g'].data = self.conv._parameters['weight_g'].data * scale_init[:, None].view(
                self.conv._parameters['weight_g'].data.size())
            self.conv._parameters['bias'].data = self.conv._parameters['bias'].data - m_init * scale_init
            return scale_init[None, :, None, None] * (x - m_init[None, :, None, None]) 


class ConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, output_padding=0, dilation=1, groups=1, bias=True, weightnorm=True, act=None, drop_prob=0.0):
        super().__init__()
        self.weightnorm = weightnorm
        self.initialized = True

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       padding=padding, output_padding=output_padding,
                                       dilation=dilation, groups=groups, bias=bias)

        self.act = nn.ELU(inplace=True) if act is not None else Identity()
        self.drop_prob = drop_prob

        if self.weightnorm:
            self.initialized = False
            self.conv = nn.utils.weight_norm(self.conv, dim=1, name="weight")

    def forward(self, input):
        if not self.initialized:
            self.init_parameters(input)
        return F.dropout(self.act(self.conv(input)), p=self.drop_prob, training=True)

    def init_parameters(self, x, init_scale=0.05, eps=1e-8):
        self.initialized = True
        if self.weightnorm:
            # initial values
            self.conv._parameters['weight_v'].data.normal_(mean=0, std=init_scale)
            self.conv._parameters['weight_g'].data.fill_(1.)
            self.conv._parameters['bias'].data.fill_(0.)
            init_scale = .01
            # data dependent init
            x = self.conv(x)
            t = x.view(x.size()[0], x.size()[1], -1)
            t = t.permute(0, 2, 1).contiguous()
            t = t.view(-1, t.size()[-1])
            m_init, v_init = torch.mean(t, 0), torch.var(t, 0)
            scale_init = init_scale / torch.sqrt(v_init + eps)

            self.conv._parameters['weight_g'].data = self.conv._parameters['weight_g'].data * scale_init[None,:].view(
                self.conv._parameters['weight_g'].data.size())
            self.conv._parameters['bias'].data = self.conv._parameters['bias'].data - m_init * scale_init
            return scale_init[None, :, None, None] * (x - m_init[None, :, None, None])


# ----- Up and Down Sampling -----

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, drop_prob=0.0):
        super().__init__()
        self.core_nn = nn.Sequential(
            Conv2d(in_channels, out_channels,
                   kernel_size=3, stride=2, padding=1, drop_prob=drop_prob)
        )

    def forward(self, input):
        return self.core_nn(input)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, drop_prob=0.0):
        super().__init__()
        self.core_nn = nn.Sequential(
            ConvTranspose2d(in_channels, out_channels,
                            kernel_size=3, stride=2, padding=1, 
                            output_padding=1, drop_prob=drop_prob)
        )

    def forward(self, input):
        return self.core_nn(input)


# ----- Gated/Attention Blocks -----

class CALayer(nn.Module):
    """
    ChannelWise Gated Layer.
    """
    def __init__(self, channel, reduction=8, drop_prob=0.0):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.ca_block = nn.Sequential(
            Conv2d(channel, channel // reduction, 
                   kernel_size=1, stride=1, padding=0, drop_prob=drop_prob),
            Conv2d(channel // reduction, channel,
                   kernel_size=1, stride=1, padding=0, act=None, drop_prob=drop_prob),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca_block(y)
        return x * y


# ----- DenseNets -----

class DenseNetBlock(nn.Module):
    def __init__(self, inplanes, growth_rate, drop_prob=0.0):
        super().__init__()
        self.dense_block = nn.Sequential(
            Conv2d(inplanes, 4 * growth_rate,
                   kernel_size=1, stride=1, padding=0, drop_prob=drop_prob),
            Conv2d(4 * growth_rate, growth_rate,
                   kernel_size=3, stride=1, padding=1, drop_prob=drop_prob, act=None)
        )

    def forward(self, input):
        y = self.dense_block(input)
        y = torch.cat([input, y], dim=1)
        return y


class DenseNetLayer(nn.Module):
    def __init__(self, inplanes, growth_rate, steps, drop_prob=0.0):
        super().__init__()
        self.activation = nn.ELU(inplace=True)

        net = []
        for step in range(steps):
            net.append(DenseNetBlock(inplanes, growth_rate, drop_prob=drop_prob))
            net.append(self.activation)
            inplanes += growth_rate

        net.append(CALayer(inplanes, drop_prob=drop_prob))
        self.core_nn = nn.Sequential(*net)

    def forward(self, input):
        return self.core_nn(input)


class DenselyNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, growth_rate, steps, blocks, act=None, drop_prob=0.0):
        super().__init__()
        # downscale block
        net = []
        for i in range(blocks):
            net.append(DenseNetLayer(in_channels, growth_rate, steps, drop_prob=drop_prob))
            in_channels = in_channels + growth_rate * steps

        # output layer
        net.append(Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, act=None))

        self.core_nn = nn.Sequential(*net)            

    def forward(self, input):
        return self.core_nn(input)


class DenselyEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, growth_rate, steps, scale_factor, drop_prob=0.0):
        super().__init__()
        # downscale block
        net = []
        for i in range(scale_factor):
            net.append(DenseNetLayer(in_channels, growth_rate, steps, drop_prob=drop_prob))
            in_channels = in_channels + growth_rate * steps
            net.append(Downsample(in_channels, 2*in_channels, drop_prob=drop_prob))
            in_channels *= 2
            growth_rate *= 2

        # output block
        net.append(DenseNetLayer(in_channels, growth_rate, steps, drop_prob=drop_prob))
        in_channels = in_channels + growth_rate * steps

        # output layer
        net.append(Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, act=None))

        self.core_nn = nn.Sequential(*net)

    def forward(self, input):
        return self.core_nn(input)


class DenselyDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, growth_rate=16, steps=3, scale_factor=2, drop_prob=0.0):
        super().__init__()
        # upsample block
        net = []
        for i in range(scale_factor):
            net.append(DenseNetLayer(in_channels, growth_rate, steps, drop_prob=drop_prob))
            in_channels = in_channels + growth_rate * steps
            net.append(Upsample(in_channels, in_channels//2, drop_prob=drop_prob))
            in_channels = in_channels//2
            growth_rate = growth_rate//2

        # output block
        net.append(Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, act=None))

        self.core_nn = nn.Sequential(*net)

    def forward(self, x):
        return self.core_nn(x)


if __name__ == "__main__":
    pass
