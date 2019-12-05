import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import imageio
import numpy as np
import cv2
from color_op import *

import torch
from torch.optim.optimizer import Optimizer, required

from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1,alpha=1.6):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        
        # self.alpha=self.alpha
        self.register_buffer('alpha', torch.FloatTensor([alpha]))
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")
        
        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, self.alpha*(w / sigma.expand_as(w)))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

# define SpectralNorm

class Autoencoder(nn.Module):
    def __init__(self,alpha=2.3):
        super(Autoencoder, self).__init__()
        channel_num = 32
        in_ = 3
        self.conv1 = SpectralNorm(nn.Conv2d(in_channels=in_, out_channels=channel_num, kernel_size=1, stride=1, bias=True,padding=0),alpha=alpha) # default lipschtiz coeficient is 1.5
        self.relu1 = nn.Tanh()
        self.conv2 = SpectralNorm(nn.Conv2d(in_channels=channel_num, out_channels=channel_num, kernel_size=1, stride=1, bias=True,padding=0),alpha=alpha)
        self.relu2 = nn.Tanh()
        self.conv3 = SpectralNorm(nn.Conv2d(in_channels=channel_num, out_channels=channel_num, kernel_size=1, stride=1, bias=True,padding=0),alpha=alpha)
        self.relu3 = nn.Tanh()
        self.conv4 = SpectralNorm(nn.Conv2d(in_channels=channel_num, out_channels=1, kernel_size=1, stride=1, bias=True,padding=0),alpha=alpha)
        self.relu4 = nn.Tanh()  #tanh fit for value
        

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu1(res)
        res = self.conv2(res)
        res = self.relu2(res)
        res = self.conv3(res)
        res = self.relu3(res)
        res = self.conv4(res)
        res = self.relu4(res)
        return res

# define AE model




class ColorModel(nn.Module):
    '''
    input should be B, C, H, W and the value should between [0, 1]
    output is B, C, H, W and the RGB is between [0, 1]
    '''


    def __init__(self,alpha=2.3):
        super(ColorModel, self).__init__()
        self.A = Autoencoder(float(alpha))
        self.B = Autoencoder(float(alpha))
        self.rgb = lab2rgb()
        self.lab = rgb2lab()
        

    def forward(self,x,y):# input rgb [0,1]

        lab_x = self.lab(x)
        a = self.A(lab_x)
        b = self.B(lab_x)
        if self.training:
            return a, b
        else:
            lab_y = self.lab(y)
            l = lab_y[:, :1, :] #select axis L from target
            lab = torch.cat((l, a, b), 1) # concate predict value and target's L
            rgb = self.rgb(lab) # convert to rgb
            return rgb # output rgb [0, 1]

