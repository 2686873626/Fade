import sys

import numpy as np
import torch
import torch.nn.functional as F
import random
from torchvision.ops import RoIAlign, deform_conv2d
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange
from pytorch_lightning import loggers as pl_loggers

from .utils import get_basic_patterns, SepConv
from .features import Xception_Fea, Swin_Fea
from .f3net import F3Net

num_samples = 8
num_nodes = 7
record = {}


class ConvIter(nn.Module):
    def __init__(self, dim=512, num_head=4) -> None:
        super().__init__()
        self.num_head = num_head
        self.update = SepConv(dim, dim, padding=1)

    def forward(self, x, A):
        """ConvGCN iteration

        Args:
            x (Tensor): B * num_head * (T * N + 1) * (512 / num_head) * H * W
            A (Tensor): B * num_head * (T * N + 1) * (T * N + 1)

        Returns:
            x (Tensor): B * num_head * (T * N + 1) * (512 / num_head) * H * W
        """
        # aggregation
        # x(n,c,i,j) = ∑_m A(n,m) * x(m,c,i,j)
        # x -- B * num_head * (T * N + 1) * (512 / num_head) * H * W
        x = einsum('b h n m, b h m c i j -> b h n c i j', A, x)

        # update
        B, H, N, C, h, w = x.shape
        # x -- B * (T * N + 1) * 512 * H * W
        x = rearrange(x, 'b h n c i j -> (b n) (h c) i j')
        x = self.update(x)
        x = rearrange(x, '(b n) (h c) i j -> b h n c i j', n=N, c=C)

        return x


class ActionGCN(nn.Module):
    def __init__(self, amplified=False):
        super().__init__()

        self.layer1 = ConvIter()
        self.layer2 = ConvIter()
        self.amp = amplified

    def forward(self, x, A):
        """Action GCN Forward

        Args:
            x (FloatTensor): B * num_head * (T * N + 1) * (512 / num_head) * H * W
            A (FloatTensor): num_head * (T * N + 1) * (T * N + 1)

        Returns:
            FloatTensor: x after massage passing
            x (Tensor): B * num_head * (T * N + 1) * (512 / num_head) * H * W
        """

        B = x.shape[0]
        # A -- B * num_head * (T * N + 1) * (T * N + 1)
        if not self.amp:
            H, M, N = A.shape
            A = torch.broadcast_to(A, (B, H, M, N))
        x = self.layer1(x, A)
        x = self.layer2(x, A)
        # node -> whole face node -> master node

        return x


class ContentGCN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = ConvIter()
        self.layer2 = ConvIter()

        self.pool = nn.Sequential(
            Rearrange('b h n d i j -> (b h n) d i j'),
            nn.AdaptiveMaxPool2d(1),
            Rearrange('(b h n) d i j -> b h n (d i j)', h=4, n=num_nodes * num_samples + 1)
        )
        nn.AdaptiveMaxPool2d(1)

        self.alpha1 = nn.Parameter(torch.tensor(1.0))
        self.alpha2 = nn.Parameter(torch.tensor(1.0))

        self.bias1 = nn.Parameter(torch.ones(1 + num_samples * num_nodes, 1 + num_samples * num_nodes))
        self.bias2 = nn.Parameter(torch.ones(1 + num_samples * num_nodes, 1 + num_samples * num_nodes))

    def forward(self, x):
        """Content GCN Forward

        Args:
            x (Tensor): BxHxNxCxhxw

        Returns:
            Tensor: B x H x N x C x h x w
        """
        vec = self.pool(x).squeeze(1)  # B x H x N x C
        vec_norm = F.normalize(vec, dim=-1)  # B x H x N x C
        A = vec_norm @ (vec_norm.transpose(-1, -2))  # B x H x N x N
        A = torch.softmax(A / self.alpha1 + self.bias1, -1)  # B x H x N x N
        x = self.layer1(x, A)  # B x H x N x C x h x w

        vec = self.pool(x).squeeze(1)
        vec_norm = F.normalize(vec, dim=-1)
        A = vec_norm @ (vec_norm.transpose(-1, -2))
        A = torch.softmax(A / self.alpha2 + self.bias2, -1)
        x = self.layer2(x, A)

        return x


class ReverseContentGCN(nn.Module):
    def __init__(self, dim=512, topk=3):
        super().__init__()
        self.layer_inter_frame = ConvIter()
        self.layer_to_master_1 = ConvIter()
        self.layer_reverse = ConvIter()
        self.layer_to_master_2 = ConvIter()
        self.topk = topk

        self.transform = SepConv(dim, dim, padding=1)

        self.pooling = nn.AdaptiveMaxPool2d(1)

        self.alpha_inter_frame = nn.Parameter(torch.tensor(1.0))
        self.alpha_to_master_1 = nn.Parameter(torch.tensor(1.0))
        self.alpha_reverse = nn.Parameter(torch.tensor(1.0))
        self.alpha_to_master_2 = nn.Parameter(torch.tensor(1.0))

        self.bias_inter_frame = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        self.bias_to_master_1 = nn.Parameter(torch.zeros(1, 1 + num_samples * num_nodes))
        self.bias_reverse = nn.Parameter(torch.zeros(num_nodes, self.topk * num_nodes))
        self.bias_to_master_2 = nn.Parameter(torch.zeros(1, 1 + num_samples * num_nodes))

    def forward(self, x):
        """
        Reverse Content GCN Forward

        Args:
            x (Tensor): B x Head x Node x C x h x w
        Returns:
            Tensor: B x Head x Node x C x h x w
        """
        # separate master node and normal nodes
        normal_x = x[:, :, 1:, :, :, :].clone()  # B x H x (T x N) x C x h x w
        master_x = x[:, :, 0, :, :, :].clone().unsqueeze(2)  # B x H x 1 x C x h x w
        # return master_x.squeeze()

        # Graph GCN in frames
        normal_x = rearrange(normal_x, 'b h (t d) c i j -> (b t) h d c i j', t=num_samples,
                             d=num_nodes)  # (B x T) x H x N x C x h x w
        vec = self.pool(normal_x).squeeze()  # (B x T) x H x N x C
        vec_norm = F.normalize(vec, dim=-1)  # (B x T) x H x N x C
        A = vec_norm @ (vec_norm.transpose(-1, -2))  # (B x T) x H x N x N
        A = torch.softmax(A / self.alpha_inter_frame + self.bias_inter_frame, -1)  # (B x T) x H x N x N
        normal_x = self.layer_inter_frame(normal_x, A)  # (B x T) x H x N x C x h x w

        # update master node
        temp_normal_x = rearrange(normal_x, '(b t) h d c i j -> b h (t d) c i j', t=num_samples,
                                  d=num_nodes)  # B x H x (T x N) x C x h x w
        all_node = torch.cat([master_x, temp_normal_x], dim=2)  # B x H x (T x N + 1) x C x h x w
        vec = self.pool(all_node).squeeze()  # B x H x (T x N + 1) x C
        vec_norm = F.normalize(vec, dim=-1)  # B x H x (T x N + 1) x C
        master_vec = vec_norm[:, :, 0, :].unsqueeze(2)  # B x H x 1 x C
        A = master_vec @ (vec_norm.transpose(-1, -2))  # B x H x 1 x (T x N + 1)
        A = torch.softmax(A / self.alpha_to_master_1 + self.bias_to_master_1, -1)  # B x H x 1 x (T x N + 1)
        master_x = self.layer_to_master_1(all_node, A)  # B x H x 1 x C x h x w

        # (B x T) x H x N x C x h x w
        B, H, N, C, h, w = normal_x.shape
        normal_x = rearrange(normal_x, 'b h n c i j -> (b n) (h c) i j')
        normal_x = self.transform(normal_x)
        normal_x = rearrange(normal_x, '(b n) (h c) i j -> b h n c i j', n=N, c=C)

        # Get 3 Lowest Similarity Nodes
        temp_normal_x = rearrange(normal_x, '(b t) h d c i j -> b h t d c i j', t=num_samples,
                                  d=num_nodes)  # B x H x T x N x C x h x w
        main_normal_x = temp_normal_x[:, :, :, -1, :, :, :]  # B x H x T x C x h x w (main node)
        main_vec = self.pool(main_normal_x).squeeze()  # B x H x T x C
        main_vec_norm = F.normalize(main_vec, dim=-1)  # B x H x T x C
        A = main_vec_norm @ (main_vec_norm.transpose(-1, -2))  # B x H x T x T
        indices = torch.topk(A, self.topk, dim=-1, largest=False, sorted=False).indices  # B x H x T x 3

        B, H, T, N, C, h, w = temp_normal_x.shape
        select_normal_x = temp_normal_x.unsqueeze(3)  # B x H x T x 1 x N x C x h x w
        select_normal_x = select_normal_x.expand(-1, -1, -1, T, -1, -1, -1, -1)  # B x H x T x T x N x C x h x w
        select_normal_x = torch.gather(select_normal_x, 3,
                                       indices.view(B, H, T, self.topk, 1, 1, 1, 1).expand(-1, -1, -1, -1, N, C, h,
                                                                                           w))  # B x H x T x 3 x N x C x h x w
        different_nodes = rearrange(select_normal_x, 'b h t n d c i j -> (b t) h (n d) c i j', n=self.topk,
                                    d=N)  # (B x T) x H x (3 x N) x C x h x w
        # different_nodes = self.lowest_gather(temp_normal_x, indices)  # (B x T) x H x (4 x N) x C x h x w

        # reverse sparse GCN
        vec_diff = self.pool(different_nodes).squeeze()  # (B x T) x H x (3 x N) x C
        vec_diff_norm = F.normalize(vec_diff, dim=-1)  # (B x T) x H x (3 x N) x C
        vec_ori = self.pool(normal_x).squeeze()  # B x H x T x N x C
        vec_ori_norm = F.normalize(vec_ori, dim=-1)  # (B x T) x H x N x C
        A = vec_ori_norm @ (vec_diff_norm.transpose(-1, -2))  # (B x T) x H x N x (3 x N)
        A = -1 * A  # Reverse !!!!!!!!!!!!!
        A = torch.softmax(A / self.alpha_reverse + self.bias_reverse, -1)  # (B x T) x H x N x (3 x N)
        normal_x = self.layer_reverse(different_nodes, A) + normal_x  # (B x T) x H x N x C x h x w  Skip Connection

        # update master node
        temp_normal_x = rearrange(normal_x, '(b t) h d c i j -> b h (t d) c i j', t=num_samples,
                                  d=num_nodes)  # B x H x (T x N) x C x h x w
        all_node = torch.cat([master_x, temp_normal_x], dim=2)  # B x H x (T x N + 1) x C x h x w
        vec = self.pool(all_node).squeeze()  # B x H x (T x N + 1) x C
        vec_norm = F.normalize(vec, dim=-1)  # B x H x (T x N + 1) x C
        master_vec = vec_norm[:, :, 0, :].unsqueeze(2)  # B x H x 1 x C
        A = master_vec @ (vec_norm.transpose(-1, -2))  # B x H x 1 x (T x N + 1)
        A = torch.softmax(A / self.alpha_to_master_2 + self.bias_to_master_2, -1)  # B x H x 1 x (T x N + 1)
        master_x = self.layer_to_master_2(all_node, A).squeeze()  # B x H x C x h x w

        return master_x  # B x H x C x h x w

    def pool(self, x):
        '''
        :param x: B x H x N x C x h x w
        :return: B x H x T x N x C
        '''
        B, H, N, C, h, w = x.shape
        x = rearrange(x, 'b h n d i j -> (b h n) d i j')
        x = self.pooling(x)
        x = rearrange(x, '(b h n) d i j -> b h n (d i j)', h=H, n=N)
        return x


class Amplified_PatternMixer(nn.Module):
    def __init__(self, num_basic=5, num_frame=8, _num_nodes=7, num_mixed=4, gaussian=True):
        super().__init__()
        self.pattern_mixer = PatternMixer(num_basic=num_basic, num_frame=num_frame, _num_nodes=_num_nodes,
                                          num_mixed=num_mixed, gaussian=gaussian)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.alpha = nn.Parameter(torch.ones(num_mixed).view(1, num_mixed, 1, 1))

    def forward(self, mat, x):
        '''
        :param mat (Tensor): num_basic x num_nodes x num_nodes
        :param x (Tensor): [B x (T x N + 1)] x C x H x W
        '''
        mixed_pattern = self.pattern_mixer(mat)  # num_mixed x (1 + num_nodes * num_frame) x (1 + num_nodes * num_frame)
        x = self.pool(x).squeeze()  # [B x (T x N + 1)] x C
        x = rearrange(x, '(b t) c -> b t c', t=num_samples * num_nodes + 1)  # B * (T * N + 1) x C
        x = torch.norm(x, dim=-1)  # B x (T x N + 1)
        x = x.unsqueeze(1)  # B x 1 x (T * N + 1)
        x = x.repeat(1, num_samples * num_nodes + 1, 1)  # B x (T * N + 1) x (T * N + 1)
        x = x.unsqueeze(1)  # B x 1 x (T * N + 1) x (T * N + 1)
        x = x.repeat(1, mixed_pattern.shape[0], 1, 1)  # B x num_mixed x (T * N + 1) x (T * N + 1)
        H, M, N = mixed_pattern.shape
        mixed_pattern = torch.broadcast_to(mixed_pattern,
                                           (x.shape[0], H, M, N))  # B x num_mixed x (T * N + 1) x (T * N + 1)
        return mixed_pattern + self.alpha * x


class PatternMixer(nn.Module):
    def __init__(self, num_basic=5, num_mixed=4, num_frame=8, _num_nodes=7, gaussian=False):
        super().__init__()
        self.num_basic = num_basic
        self.num_mixed = num_mixed
        self.num_frame = num_frame
        self.num_nodes = _num_nodes
        self.gaussian = gaussian

        self.pattern_mixer = nn.ModuleList()
        for _ in range(self.num_mixed):
            # input: num_nodes x ((2 * num_frame - 1) x num_nodes) x num_basic
            self.pattern_mixer.append(
                nn.Sequential(
                    nn.Linear(num_basic, 1),  # num_nodes x ((2 * num_frame - 1) x num_nodes) x 1
                    Rearrange('n m h-> h n m'),  # 1 x num_nodes x ((2 * num_frame - 1) x num_nodes)
                    nn.ReLU(),
                )
            )

        # gaussian prior
        if self.gaussian:
            self.sigma = nn.Parameter(torch.ones(num_mixed, num_basic, 1))
            distance = torch.arange(num_frame * 2 - 1).float()
            distance = torch.abs(distance - (num_frame - 1))
            self.register_buffer('distance', distance)
        else:
            self.temporal_expansion = nn.Parameter(torch.randn(num_mixed, num_basic, num_frame * 2 - 1))

        mixed_mat = torch.zeros(
            self.num_mixed,
            1 + self.num_nodes * self.num_frame,
            1 + self.num_nodes * self.num_frame
        )
        # mixed_mat = torch.eye()
        mixed_mat[:, :, 0] = 0.5  # every node to the master node
        mixed_mat[:, 0, ::num_nodes] = 1.0  # master node to each whole face node
        # mixed_mat[:, 0, :] = 0.5
        # mixed_mat[:, ::num_nodes, 0] = 1.0 # each whole face node to master node
        self.mixed_mat = nn.Parameter(mixed_mat)
        self.register_buffer('degree_mat', torch.zeros_like(mixed_mat))

    def forward(self, mat):
        """Mix the basic pattern

        Args:
            mat (Tensor): num_basic x num_nodes x num_nodes

        Returns:
            Tensor: num_mixed x (1 + num_nodes * num_frame) x (1 + num_nodes * num_frame)
        """
        mat_list = []
        if self.gaussian:
            self.temporal_expansion = 1.0 / (np.sqrt(2 * np.pi) * self.sigma) * torch.exp(
                -self.distance ** 2 / (2 * self.sigma ** 2))

        for i in range(self.num_mixed):
            cur_expansion = self.temporal_expansion[i]
            cur_expansion = torch.sigmoid(cur_expansion).squeeze(0)
            # multiply the expansion on the channel dim for each basic pattern
            # num_basic x (2 * num_frame - 1) x num_nodes x num_nodes
            input_mat = einsum('b t, b n m -> b t n m', cur_expansion, mat)
            input_mat = rearrange(input_mat,
                                  'b t n m -> n (t m) b')  # num_nodes x ((2 * num_frame - 1) x num_nodes) x num_basic
            mat_list.append(self.pattern_mixer[i](input_mat))  # 1 x num_nodes x ((2 * num_frame - 1) x num_nodes)

        mat = torch.cat(mat_list)  # num_mixed x num_nodes x ((2 * num_frame - 1) x num_nodes)

        mixed_mat = self.mixed_mat.clone()

        for i in range(num_samples):
            # sliding window
            # equal weight for specific units with equal time interval
            mixed_mat[:, (1 + i * self.num_nodes):(1 + (i + 1) * self.num_nodes), 1:] \
                += mat[..., (self.num_nodes * (num_samples - 1 - i)):(self.num_nodes * (num_samples * 2 - 1 - i))]

        # 当前帧的总节点不与其他帧的其他节点相连
        for i in range(num_samples):
            for j in range(num_samples):
                if i != j:
                    # whole face node to whole face node
                    value = mixed_mat[:, (1 + i) * num_nodes, (1 + j) * num_nodes].clone()
                    # whole face node to nodes of other frame = 0
                    mixed_mat[:, (1 + i) * num_nodes, 1 + j * num_nodes: 1 + (1 + j) * num_nodes] = 0.0
                    # restore the value
                    mixed_mat[:, (1 + i) * num_nodes, (1 + j) * num_nodes] = value

        deg_mat = self.degree_mat.clone()
        for i in range(1 + num_samples * num_nodes):
            deg_mat[:, i, i] = torch.pow(mixed_mat[:, i].sum(dim=1).clamp_(min=1.0), -0.5)

        norm = deg_mat
        ret = norm @ mixed_mat @ norm
        # torch.save(ret.cpu(), 'mix_pattern.pt')
        return ret


class MultiDepGraphModule(nn.Module):
    def __init__(self, input_dim=2048, num_head=4):
        super().__init__()
        self.num_head = num_head

        # basic_patterns -> num_basic * num_nodes * num_nodes
        basic_patterns = get_basic_patterns(num_nodes=num_nodes, num_basic=5)
        self.register_buffer('basic_patterns', basic_patterns)
        self.pattern_mixer = Amplified_PatternMixer(num_basic=5, num_frame=num_samples, _num_nodes=num_nodes,
                                                    num_mixed=self.num_head, gaussian=False)

        self.master_node = nn.Parameter(torch.randn(1, 1, input_dim, 3, 3))

        self.gcn1 = ActionGCN(amplified=True)
        self.gcn2 = ReverseContentGCN()

        self.proj1 = SepConv(input_dim, 512, padding=1)
        self.proj2 = SepConv(input_dim, 512, padding=1)

        self.fusion = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv2d(512, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1)
        )

    def forward(self, x):
        """MDGM Forward

        Args:
            x (Tensor): B * (T * N) * C * H * W    N -- num_nodes
        Returns:
            Tensor: BxD
        """
        B = x.shape[0]
        # master_node -- B, 1, C, 3, 3
        x = torch.cat([self.master_node.repeat(B, 1, 1, 1, 1), x], dim=1)

        # B * (T * N + 1) * C * H * W -> [B * (T * N + 1)] * C * H * W
        x = rearrange(x, 'b n c h w -> (b n) c h w')

        # Action Graph
        # mixed_patterns -- Mixed_Pattern_num * (T * N + 1) * (T * N + 1)
        # mixed_patterns = self.pattern_mixer(self.basic_patterns)
        mixed_patterns = self.pattern_mixer(self.basic_patterns, x)  # amplify the patterns
        print(mixed_patterns.shape)

        # x1 -- [B * (T * N + 1)] * 512 * H * W
        x1 = self.proj1(x)
        # x1 -- B * num_head * (T * N + 1) * (512 / num_head) * H * W
        x1 = rearrange(x1, '(b n) (h d) i j -> b h n d i j', b=B, h=self.num_head)
        x1 = self.gcn1(x1, mixed_patterns)

        # x1 -- B * (T * N + 1) * 512 * H * W
        x1 = rearrange(x1, 'b h n d i j -> b n (h d) i j', h=self.num_head)
        # master node x1 -- B * 512 * H * W
        x1 = x1[:, 0]

        # Content Graph
        # x2 -- [B * (T * N + 1)] * 512 * H * W
        x2 = self.proj2(x)
        # x2 -- B * num_head * (T * N + 1) * (512 / num_head) * H * W
        x2 = rearrange(x2, '(b n) (h d) i j -> b h n d i j', b=B, h=self.num_head)
        x2 = self.gcn2(x2)  # B x H x C x h x w

        # Reverse Content Branch
        x2 = rearrange(x2, 'b h c i j -> b (h c) i j', h=self.num_head)

        # Content Branch
        # x2 = rearrange(x2, 'b h n d i j -> b n (h d) i j', h=self.num_head)
        # # master node x2 -- B * 512 * H * W
        # x2 = x2[:, 0]

        # Fusion
        x = self.fusion(torch.cat([x1, x2], dim=1))

        return x


class V2G(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = Xception_Fea()

        # deformable conv
        # self.deform_kernel_size = 3
        # self.deform_group = 1
        # self.deform_weight = nn.Parameter(torch.randn(2048, 2048 // self.deform_group, self.deform_kernel_size, self.deform_kernel_size))
        # self.deform_bias = nn.Parameter(torch.randn(2048))
        # self.deform_offset = nn.Conv2d(2048, 2 * self.deform_group * self.deform_kernel_size * self.deform_kernel_size, self.deform_kernel_size, padding=1)

        # self.gcn = MultiDepGraphModule(input_dim=2048, num_head=4)
        # self.features = Swin_Fea()
        # self.gcn = MultiDepGraphModule(input_dim=1024)

        # self.features = F3Net()
        # self.features.load_state_dict(torch.load('init_weights/f3net-LQ.ckpt', map_location='cpu'))

        # for name, m in self.features.named_modules():
        #     if 'conv1' in name:
        #         m.requires_grad_(False)
        #     elif 'conv2' in name:
        #         m.requires_grad_(False)
        # elif 'block1' in name:
        #     m.requires_grad_(False)
        # del self.features.lfs_excep
        # del self.features.lfs_head
        # del self.features.mix_block12
        # del self.features.mix_block7

        # self.gcn = MultiDepGraphModule(input_dim=4096)
        self.gcn = MultiDepGraphModule(input_dim=2048, num_head=4)

        self.roi = RoIAlign((3, 3), 1.0, -1, aligned=True)  # (3,3)

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Softmax(1)
        )

    def forward(self, x, rois):
        """V2G forward

        Args:
            x (FloatTensor): (B * T) * C * H * W
            rois (FloatTensor): (B * T * num_nodes) * 5, areas including whole face, left eye, right eye, left_cheek, right_cheek, nose and mouth
                                [frame idx, x1, y1, x2, y2]  frame idx of same video is same (num_nodes of idx is same)

        Returns:
            FloatTensor: video-level predictions 
        """
        B = x.shape[0]
        x = self.features(x)

        # Deformable Conv
        # offset = self.deform_offset(x)
        # x = deform_conv2d(x, offset, self.deform_weight, self.deform_bias, padding=(1,1))

        # Adaptive RoI
        # height, width = x.shape[-2:]
        # rois[:, 1:] = rois[:, 1:] * torch.tensor([width, height, width, height], dtype=rois.dtype, device=rois.device)

        x = self.roi(x, rois)  # (B * T * num_nodes) * C * 3 * 3

        _, C, H, W = x.shape

        x = x.view(B // num_samples, num_samples * num_nodes, C, H, W)  # B * (T * num_nodes) * C * 3 * 3

        # if not self.training:
        #     b, n = x.shape[0], x.shape[1]
        #     for i in range(b):    
        #         idx_drop = list(range(0, num_nodes-1))
        #         k=2
        #         idx_drop = list[:k]
        #         for j in range(num_samples):
        #             for idx in idx_drop:
        #                 x[i, (j+1)*num_nodes-1] -= x[i, idx+j*num_nodes]/(num_nodes-1)
        #                 x[i, idx+j*num_nodes]=0.

        x = self.gcn(x)

        return self.classifier(x)
