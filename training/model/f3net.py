import os
import sys
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from pretrainedmodels import xception


class F3Net(nn.Module):
    """
    Implementation is mainly referenced from https://github.com/yyk-wew/F3Net
    """
    def __init__(self,
                 num_classes: int=2, 
                 img_width: int=288, 
                 img_height: int=288, 
                 LFS_window_size: int=10, 
                 LFS_M: int=6) -> None:
        super(F3Net, self).__init__()
        assert img_width == img_height
        self.img_size = img_width
        self.num_classes = num_classes
        self._LFS_window_size = LFS_window_size
        self._LFS_M = LFS_M
        
        
        self.fad_head = FAD_Head(self.img_size)
        self.lfs_head = LFS_Head(self.img_size, self._LFS_window_size, self._LFS_M)
        
        self.fad_excep = self._init_xcep_fad()
        self.lfs_excep = self._init_xcep_lfs()
        
        self.mix_block7 = MixBlock(c_in=728, width=19, height=19) 
        self.mix_block12 = MixBlock(c_in=1024, width=10, height=10) 
        self.excep_forwards = ['conv1', 'bn1', 'relu', 'conv2', 'bn2', 'relu', 
                               'block1', 'block2', 'block3', 'block4', 'block5', 'block6', 
                               'block7', 'block8', 'block9', 'block10' , 'block11', 'block12',
                               'conv3', 'bn3', 'relu', 'conv4', 'bn4']

         # classifier
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(4096, num_classes)
        self.dp = nn.Dropout(p=0.2)
        
    def _init_xcep_fad(self):
        fad_excep =  xception()
        conv1_data = fad_excep.conv1.weight.data
        # let new conv1 use old param to balance the network
        fad_excep.conv1 = nn.Conv2d(12, 32, 3, 2, 0, bias=False)
        for i in range(4):
            fad_excep.conv1.weight.data[:, i*3:(i+1)*3, :, :] = conv1_data / 4.0
        return fad_excep
    
    def  _init_xcep_lfs(self): 
        lfs_excep = xception()
        conv1_data = lfs_excep.conv1.weight.data
        # let new conv1 use old param to balance the network
        lfs_excep.conv1 = nn.Conv2d(self._LFS_M, 32, 3, 1, 0, bias=False)
        for i in range(int(self._LFS_M / 3)):
            lfs_excep.conv1.weight.data[:, i*3:(i+1)*3, :, :] = conv1_data / float(self._LFS_M / 3.0)
        return lfs_excep
    
    def _features(self, x_fad, x_fls):
        for forward_func in self.excep_forwards:
            x_fad = getattr(self.fad_excep, forward_func)(x_fad)
            x_fls = getattr(self.lfs_excep, forward_func)(x_fls)
            if forward_func == 'block7':
                x_fad, x_fls = self.mix_block7(x_fad, x_fls)
            if forward_func == 'block12':
                x_fad, x_fls = self.mix_block12(x_fad, x_fls)
        return torch.cat([x_fad, x_fls], dim=1)
        # return x_fad
    
    def _norm_feature(self, x):
        x = self.relu(x)
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = x.view(x.size(0), -1)
        return x
    
    def forward(self, x):
        fad_input = self.fad_head(x)
        lfs_input = self.lfs_head(x)
        # lfs_input = None
        x_cat = self._features(fad_input, lfs_input)
        # x_cat = self._norm_feature(x_cat)
        # x_drop = self.dp(x_cat)
        # logit = self.fc(x_drop)
        # return torch.softmax(logit, -1)
        return x_cat


class Filter(nn.Module):
    def __init__(self, size, 
                 band_start, 
                 band_end, 
                 use_learnable=True, 
                 norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
            # Todo
            # self.learnable = nn.Parameter(torch.rand((size, size)) * 0.2 - 0.1, requires_grad=True)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False)


    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y

class FAD_Head(nn.Module):
    def __init__(self, size):
        super(FAD_Head, self).__init__()
        # init DCT matrix
        self._DCT_all = nn.Parameter(torch.tensor(DCT_mat(size)).float(), requires_grad=False)
        self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(size)).float(), 0, 1), requires_grad=False)

        # define base filters and learnable
        # 0 - 1/16 || 1/16 - 1/8 || 1/8 - 1
        low_filter = Filter(size, 0, size // 2.82)
        middle_filter = Filter(size, size // 2.82, size // 2)
        high_filter = Filter(size, size // 2, size*2)
        all_filter = Filter(size, 0, size * 2)

        self.filters = nn.ModuleList([low_filter, middle_filter, high_filter, all_filter])

    def forward(self, x):
        # DCT
        x_freq = self._DCT_all @ x @ self._DCT_all_T    # [N, 3, 288, 288]

        # 4 kernel
        y_list = []
        for i in range(4):
            x_pass = self.filters[i](x_freq)  # [N, 3, 288, 288]
            y = self._DCT_all_T @ x_pass @ self._DCT_all    # [N, 3, 288, 288]
            y_list.append(y)
        out = torch.cat(y_list, dim=1)    # [N, 12, 288, 288]
        return out


class LFS_Head(nn.Module):
    def __init__(self, size, window_size, M):
        super(LFS_Head, self).__init__()

        self.window_size = window_size
        self._M = M

        # init DCT matrix
        self._DCT_patch = nn.Parameter(torch.tensor(DCT_mat(window_size)).float(), requires_grad=False)
        self._DCT_patch_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(window_size)).float(), 0, 1), requires_grad=False)

        self.unfold = nn.Unfold(kernel_size=(window_size, window_size), stride=2, padding=4)

        # init filters
        self.filters = nn.ModuleList([Filter(window_size, window_size * 2. / M * i, window_size * 2. / M * (i+1), norm=True) for i in range(M)])
    
    def forward(self, x):
        # turn RGB into Gray
        x_gray = 0.299*x[:,0,:,:] + 0.587*x[:,1,:,:] + 0.114*x[:,2,:,:]
        x = x_gray.unsqueeze(1)

        # rescale to 0 - 255
        x = (x + 1.) * 122.5

        # calculate size
        N, C, W, H = x.size()
        S = self.window_size
        size_after = int((W - S + 8)/2) + 1
        # assert size_after == 149

        # sliding window unfold and DCT
        x_unfold = self.unfold(x)   # [N, C * S * S, L]   L:block num
        L = x_unfold.size()[2]
        x_unfold = x_unfold.transpose(1, 2).reshape(N, L, C, S, S)  # [N, L, C, S, S]
        x_dct = self._DCT_patch @ x_unfold @ self._DCT_patch_T

        # M kernels filtering
        y_list = []
        for i in range(self._M):
            # y = self.filters[i](x_dct)    # [N, L, C, S, S]
            # y = torch.abs(y)
            # y = torch.sum(y, dim=[2,3,4])   # [N, L]
            # y = torch.log10(y + 1e-15)
            y = torch.abs(x_dct)
            y = torch.log10(y + 1e-15)
            y = self.filters[i](y)
            y = torch.sum(y, dim=[2,3,4])
            y = y.reshape(N, size_after, size_after).unsqueeze(dim=1)   # [N, 1, 149, 149]
            y_list.append(y)
        out = torch.cat(y_list, dim=1)  # [N, M, 149, 149]
        return out
    
class MixBlock(nn.Module):
    
    def __init__(self, c_in, width, height):
        super(MixBlock, self).__init__()
        self.FAD_query = nn.Conv2d(c_in, c_in, (1,1))
        self.LFS_query = nn.Conv2d(c_in, c_in, (1,1))

        self.FAD_key = nn.Conv2d(c_in, c_in, (1,1))
        self.LFS_key = nn.Conv2d(c_in, c_in, (1,1))

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

        self.FAD_gamma = nn.Parameter(torch.zeros(1))
        self.LFS_gamma = nn.Parameter(torch.zeros(1))

        self.FAD_conv = nn.Conv2d(c_in, c_in, (1,1), groups=c_in)
        self.FAD_bn = nn.BatchNorm2d(c_in)
        self.LFS_conv = nn.Conv2d(c_in, c_in, (1,1), groups=c_in)
        self.LFS_bn = nn.BatchNorm2d(c_in)

    def forward(self, x_FAD, x_LFS):
        B, C, W, H = x_FAD.size()
        assert W == H

        q_FAD = self.FAD_query(x_FAD).view(-1, W, H)    # [BC, W, H]
        q_LFS = self.LFS_query(x_LFS).view(-1, W, H)
        M_query = torch.cat([q_FAD, q_LFS], dim=2)  # [BC, W, 2H]

        k_FAD = self.FAD_key(x_FAD).view(-1, W, H).transpose(1, 2)  # [BC, H, W]
        k_LFS = self.LFS_key(x_LFS).view(-1, W, H).transpose(1, 2)
        M_key = torch.cat([k_FAD, k_LFS], dim=1)    # [BC, 2H, W]

        energy = torch.bmm(M_query, M_key)  #[BC, W, W]
        attention = self.softmax(energy).view(B, C, W, W)

        att_LFS = x_LFS * attention * (torch.sigmoid(self.LFS_gamma) * 2.0 - 1.0)
        y_FAD = x_FAD + self.FAD_bn(self.FAD_conv(att_LFS))

        att_FAD = x_FAD * attention * (torch.sigmoid(self.FAD_gamma) * 2.0 - 1.0)
        y_LFS = x_LFS + self.LFS_bn(self.LFS_conv(att_FAD))
        return y_FAD, y_LFS

# utils
def DCT_mat(size):
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m

def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j <= start else 1. for j in range(size)] for i in range(size)]

def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.