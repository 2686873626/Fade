import torch
import torch.nn.functional as F
import random
from torchvision.ops import RoIAlign
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange
from .utils import get_basic_patterns, SepConv
from .features import Xception_Fea, Swin_Fea


clip_len = 8
group_size = 7


class ConvIter(nn.Module):
    def __init__(self, dim=512, num_head=4) -> None:
        super().__init__()
        self.num_head = num_head
        self.update = SepConv(dim, dim, padding=1)

    def forward(self, x, A):
        """ConvGCN iteration

        Args:
            x (Tensor): BxHxNxCxhxw
            A (Tensor): BxHxNxN
        """        
        # aggregation
        x = einsum('b h n m, b h m c i j -> b h n c i j', A, x)
        
        # update
        B, H, N, C, h, w = x.shape
        x = rearrange(x, 'b h n c i j -> (b n) (h c) i j')
        x = self.update(x)
        x = rearrange(x, '(b n) (h c) i j -> b h n c i j', n=N, c=C)
        
        return x


class ActionGCN(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.layer1 = ConvIter()
        self.layer2 = ConvIter()

    def forward(self, x, A):
        """Action GCN Forward

        Args:
            x (FloatTensor): BxHxNxCxhxw
            A (FloatTensor): HxNxN

        Returns:
            FloatTensor: x after massage passing
        """    
        B = x.shape[0]
        H, M, N = A.shape
        A = torch.broadcast_to(A, (B, H, M, N))
        x = self.layer1(x, A)
        x = self.layer2(x, A)
        
        return x


class ContentGCN(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.layer1 = ConvIter()
        self.layer2 = ConvIter()

        self.pool = nn.Sequential(
            Rearrange('b h n d i j -> (b h n) d i j'),
            nn.AdaptiveMaxPool2d(1),
            Rearrange('(b h n) d i j -> b h n (d i j)', h=4, n=group_size*clip_len+1)
        )
        nn.AdaptiveMaxPool2d(1)

        self.alpha1 = nn.Parameter(torch.tensor(1.0))
        self.alpha2 = nn.Parameter(torch.tensor(1.0))

        self.bias1 = nn.Parameter(torch.ones(1+clip_len*group_size, 1+clip_len*group_size))
        self.bias2 = nn.Parameter(torch.ones(1+clip_len*group_size, 1+clip_len*group_size))

    def forward(self, x):
        """Content GCN Forward

        Args:
            x (Tensor): BxHxNxCxhxw

        Returns:
            Tensor: BxHxNxN
        """
        vec = self.pool(x).squeeze(1)
        vec_norm = F.normalize(vec, dim=-1)
        A = vec_norm@(vec_norm.transpose(-1, -2))
        A = torch.softmax(A/self.alpha1+self.bias1, -1)
        x = self.layer1(x, A)
        
        vec = self.pool(x).squeeze(1)
        vec_norm = F.normalize(vec, dim=-1)
        A = vec_norm@(vec_norm.transpose(-1, -2))
        A = torch.softmax(A/self.alpha2+self.bias2, -1)
        x = self.layer2(x, A)

        return x


class PatternMixer(nn.Module):
    def __init__(self, num_basic=2, num_mixed=4, num_frame=8, num_nodes=5):
        super().__init__()
        self.num_basic = num_basic
        self.num_mixed = num_mixed
        self.num_frame = num_frame
        self.num_nodes = num_nodes
        
        self.pattern_mixer = nn.ModuleList()
        for _ in range(self.num_mixed):
            self.pattern_mixer.append(
                nn.Sequential(
                    nn.Linear(num_basic, 1),
                    Rearrange('n m h-> h n m'),
                    nn.ReLU(),
                )
            )
        
        self.temporal_expansion = nn.Parameter(torch.randn(num_mixed, num_basic, num_frame*2-1))


        mixed_mat = torch.zeros(
            self.num_mixed,
            1+self.num_nodes*self.num_frame,
            1+self.num_nodes*self.num_frame
        )
        mixed_mat[:, :, 0]=0.5
        mixed_mat[:, 0, ::group_size]=1.0
        self.mixed_mat = nn.Parameter(mixed_mat)
        self.register_buffer('degree_mat', torch.zeros_like(mixed_mat))

    def forward(self, mat):
        """Mix the basic pattern

        Args:
            mat (Tensor): Hxnxn

        Returns:
            Tensor: HxNxN
        """     
        mat_list = []
        for i in range(self.num_mixed):
            cur_expansion = self.temporal_expansion[i]
            cur_expansion = torch.sigmoid(cur_expansion).squeeze(0)
            input_mat = einsum('h t, h n m -> h t n m', cur_expansion, mat)
            input_mat = rearrange(input_mat, 'h t n m -> n (t m) h')
            mat_list.append(self.pattern_mixer[i](input_mat))
            
        mat = torch.cat(mat_list)

        mixed_mat = self.mixed_mat.clone()

        for i in range(clip_len):
            mixed_mat[ :, 1+i*self.num_nodes:1+(i+1)*self.num_nodes, 1:] \
                    += mat[..., self.num_nodes*(clip_len-1-i):self.num_nodes*(clip_len*2-1-i)]

        for i in range(clip_len):
            for j in range(clip_len):
                if i!=j:
                    value = mixed_mat[:, (1+i)*group_size, (1+j)*group_size].clone()
                    mixed_mat[:, (1+i)*group_size, 1+j*group_size: 1+(1+j)*group_size] = 0.0
                    mixed_mat[:, (1+i)*group_size, (1+j)*group_size] = value

        deg_mat = self.degree_mat.clone()
        for i in range(1+clip_len*group_size):
            deg_mat[:, i, i] = torch.pow(mixed_mat[:, i].sum(dim=1).clamp_(min=1.0), -0.5)
        
        norm = deg_mat
        ret = norm@mixed_mat@norm

        # torch.save(ret.cpu(), 'mix_pattern.pt')
        
        return ret


# class PatternMixer(nn.Module):
#     def __init__(self, num_basic=5, num_mixed=4, num_frame=8, num_nodes=5):
#         super().__init__()
#         self.num_basic = num_basic
#         self.num_mixed = num_mixed
#         self.num_frame = num_frame
#         self.num_nodes = num_nodes

#         self.temp_extent = nn.Parameter(torch.randn(num_basic, num_frame*2-1), requires_grad=True)

#         self.pattern_mix = nn.Sequential(
#             nn.Linear(num_basic, num_mixed),
#             Rearrange('m n h -> h n m'),
#             nn.ReLU()
#         )

#         mixed_mat = torch.zeros(
#             self.num_mixed,
#             1+self.num_nodes*self.num_frame,
#             1+self.num_nodes*self.num_frame
#         )
#         mixed_mat[:, 0::7, 0::7]=1.0
#         self.register_buffer('mixed_mat', mixed_mat)
#         self.register_buffer('degree_mat', torch.zeros_like(mixed_mat))

#     def forward(self, mat):
#         """Mix the basic pattern

#         Args:
#             mat (Tensor): Hxnxn

#         Returns:
#             Tensor: HxNxN
#         """     
#         # temp_extent = F.relu(self.temp_extent)
#         temp_extent = torch.sigmoid(self.temp_extent)+1
#         mat = einsum('h t, h n m -> h t m n', temp_extent, mat)

#         mat = rearrange(mat, 'h t m n -> (t m) n h')
#         mat = F.relu(mat)
#         mat = self.pattern_mix(mat)

#         mixed_mat = self.mixed_mat.clone()

#         for i in range(clip_len):
#             mixed_mat[ :, 1+i*self.num_nodes:1+(i+1)*self.num_nodes, 1:] \
#                     = mat[..., self.num_nodes*(clip_len-1-i):self.num_nodes*(clip_len*2-1-i)]

#         deg_mat = self.degree_mat.clone()
#         for i in range(1+clip_len*group_size):
#             deg_mat[:, i, i] = torch.pow(mixed_mat[:, i].sum(dim=1).clamp_(min=1.0), -0.5)
        
#         norm = deg_mat
#         ret = norm@mixed_mat@norm

#         # torch.save(ret.cpu(), 'mix_pattern.pt')
        
#         return ret


class MultiDepGraphModule(nn.Module):
    def __init__(self, input_dim=2048, num_head=4):
        super().__init__()
        self.num_head = num_head

        basic_patterns = get_basic_patterns(group_size=group_size)
        
        self.register_buffer('basic_patterns', basic_patterns)
        self.master_node = nn.Parameter(torch.randn(1, 1, input_dim, 3, 3))
        
        self.pattern_mixer = PatternMixer(num_frame=clip_len, num_nodes=group_size)
        
        self.gcn1 = ActionGCN(1024//num_head)
        self.gcn2 = ContentGCN(1024//num_head)

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
            x (Tensor): BxNxCxhxw

        Returns:
            Tensor: BxD
        """        
        B = x.shape[0]
        x = torch.cat([self.master_node.repeat(B, 1, 1, 1, 1), x], dim=1)        
        
        # B, N, C, h, w = x.shape
        x = rearrange(x, 'b n c h w -> (b n) c h w')

        # Action Graph
        mixed_patterns = self.pattern_mixer(self.basic_patterns)
        
        x1 = self.proj1(x)
        x1 = rearrange(x1, '(b n) (h d) i j -> b h n d i j', b=B, h=self.num_head)
        x1 = self.gcn1(x1, mixed_patterns)

        x1 = rearrange(x1, 'b h n d i j -> b n (h d) i j', h=self.num_head)
        x1 = x1[:, 0]

        # Content Graph
        x2 = self.proj2(x)
        x2 = rearrange(x2, '(b n) (h d) i j -> b h n d i j', b=B, h=self.num_head)
        x2 = self.gcn2(x2)

        x2 = rearrange(x2, 'b h n d i j -> b n (h d) i j', h=self.num_head)
        x2 = x2[:, 0]

        # Fusion
        x = self.fusion(torch.cat([x1, x2], dim=1))

        return x

from .f3net import F3Net

class V2G(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = Xception_Fea()
        self.gcn = MultiDepGraphModule(input_dim=2048)
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
        self.gcn = MultiDepGraphModule(input_dim=2048)
        
        self.roi = RoIAlign((3, 3), 1.0, -1, aligned=True)

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
            x (FloatTensor): BxCxHxW
            rois (FloatTensor): (B*7)*5, areas including whole face, left eye, right eye, nose and mouth

        Returns:
            FloatTensor: video-level predictions 
        """      
        B = x.shape[0]  
        x = self.features(x)

        x = self.roi(x, rois)
        _, C, H, W = x.shape
        
        x = x.view(B//clip_len, clip_len*group_size, C, H, W)

        # if not self.training:
        #     b, n = x.shape[0], x.shape[1]
        #     for i in range(b):    
        #         idx_drop = list(range(0, group_size-1))
        #         k=2
        #         idx_drop = list[:k]
        #         for j in range(clip_len):
        #             for idx in idx_drop:
        #                 x[i, (j+1)*group_size-1] -= x[i, idx+j*group_size]/(group_size-1)
        #                 x[i, idx+j*group_size]=0.

        x = self.gcn(x)

        return self.classifier(x)