import torch
import torch.nn.functional as F
from torchvision.ops import RoIAlign
from torch import nn, einsum
from einops import rearrange
from einops.layers.torch import Rearrange
from .utils import get_basic_patterns, SepConv
from .features import Xception_Fea


clip_len = 8
group_size = 7


class ActionGCN(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.layer1 = nn.Linear(dim, dim, bias=False)
        self.layer2 = nn.Linear(dim, dim, bias=False)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, x, A):
        """Action GCN Forward

        Args:
            x (FloatTensor): BxHxNxD
            A (FloatTensor): BxHxNxN

        Returns:
            FloatTensor: x after massage passing
        """              
        x = self.norm1(x)
        x = self.dropout(x)
        x = self.relu(self.layer1(A@x))

        x = self.norm2(x)
        x = self.dropout(x)
        x = self.relu(self.layer2(A@x))
        
        return x


class ContentGCN(nn.Module):
    def __init__(self, dim, num_head=4):
        super().__init__()

        self.layer1 = nn.Linear(dim, dim, bias=False)
        self.layer2 = nn.Linear(dim, dim, bias=False)

        # self.to_qk1 = nn.Sequential(
        #     Rearrange('b h n d -> b n (h d)'),
        #     nn.Linear(dim*4, dim*8, bias=False),
        #     Rearrange('b n (m h d) -> m b h n d', m=2, h=num_head),
        # )
        # self.to_qk2 = nn.Sequential(
        #     Rearrange('b h n d -> b n (h d)'),
        #     nn.Linear(dim*4, dim*8, bias=False),
        #     Rearrange('b n (m h d) -> m b h n d', m=2, h=num_head),
        # )
                
        # self.to_qkv1 = nn.Sequential(
        #     Rearrange('b h n d -> b n (h d)'),
        #     nn.Linear(dim*num_head, dim*num_head*3, bias=False),
        #     Rearrange('b n (m h d) -> m b h n d', m=3, h=num_head),
        # )
        # self.to_qkv2 = nn.Sequential(
        #     Rearrange('b h n d -> b n (h d)'),
        #     nn.Linear(dim*num_head, dim*num_head*3, bias=False),
        #     Rearrange('b n (m h d) -> m b h n d', m=3, h=num_head),
        # )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

        self.alpha1 = nn.Parameter(torch.tensor(1.0))
        self.alpha2 = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        """Content GCN Forward

        Args:
            x (Tensor): BxHxNxD

        Returns:
            Tensor: BxHxNxN
        """
        # q, k, v = self.to_qkv1(x)
        q, k, v = x.clone(), x.clone(), x.clone()
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(k, dim=-1)
        A = q_norm@(k_norm.transpose(-1, -2))
        A = torch.softmax(A/self.alpha1.clamp(min=1e-2), dim=-1)
        # A = self.dropout(A)
        x = self.relu(self.layer1(A@v)+x)

        # q, k, v = self.to_qkv2(x)
        q, k, v = x.clone(), x.clone(), x.clone()
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(k, dim=-1)
        A = q_norm@(k_norm.transpose(-1, -2))
        A = torch.softmax(A/self.alpha2.clamp(min=1e-2), dim=-1)
        # A = self.dropout(A)
        x = self.relu(self.layer2(A@v)+x)
        
        return x


class PatternMixer(nn.Module):
    def __init__(self, num_basic=5, num_mixed=4, num_frame=8, num_nodes=5):
        super().__init__()
        self.num_basic = num_basic
        self.num_mixed = num_mixed
        self.num_frame = num_frame
        self.num_nodes = num_nodes

        self.temp_extent = nn.Parameter(torch.randn(num_basic, num_frame*2-1), requires_grad=True)

        self.pattern_mix = nn.Sequential(
            nn.Linear(num_basic, num_mixed),
            Rearrange('m n h -> h n m'),
            nn.ReLU()
        )

        mixed_mat = torch.zeros(
            self.num_mixed,
            1+self.num_nodes*self.num_frame,
            1+self.num_nodes*self.num_frame
        )
        mixed_mat[:, 0::7, 0::7]=1.0
        self.register_buffer('mixed_mat', mixed_mat)
        self.register_buffer('degree_mat', torch.zeros_like(mixed_mat))

    def forward(self, mat):
        """Mix the basic pattern

        Args:
            mat (Tensor): Hxnxn

        Returns:
            Tensor: HxNxN
        """     
        # temp_extent = F.relu(self.temp_extent)
        temp_extent = torch.sigmoid(self.temp_extent)+1
        mat = einsum('h t, h n m -> h t m n', temp_extent, mat)

        mat = rearrange(mat, 'h t m n -> (t m) n h')
        mat = F.relu(mat)
        mat = self.pattern_mix(mat)

        mixed_mat = self.mixed_mat.clone()

        for i in range(clip_len):
            mixed_mat[ :, 1+i*self.num_nodes:1+(i+1)*self.num_nodes, 1:] \
                    = mat[..., self.num_nodes*(clip_len-1-i):self.num_nodes*(clip_len*2-1-i)]

        deg_mat = self.degree_mat.clone()
        for i in range(1+clip_len*group_size):
            deg_mat[:, i, i] = torch.pow(mixed_mat[:, i].sum(dim=1).clamp_(min=1.0), -0.5)
        
        norm = deg_mat
        ret = norm@mixed_mat@norm

        # torch.save(ret.cpu(), 'mix_pattern.pt')
        
        return ret


class MultiDepGraphModule(nn.Module):
    def __init__(self, num_head=4):
        super().__init__()
        self.num_head = num_head

        basic_patterns = get_basic_patterns(group_size=group_size)
        
        self.register_buffer('basic_patterns', basic_patterns)
        self.master_node = nn.Parameter(torch.randn(1, 1, 2048))
        
        self.pattern_mixer = PatternMixer(num_nodes=group_size)
        
        self.gcn1 = ActionGCN(1024//num_head)
        self.gcn2 = ContentGCN(1024//num_head)

        self.mlp1 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            Rearrange('b n (h d) -> b h n d', h=num_head)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            Rearrange('b n (h d) -> b h n d', h=num_head)
        )

        # self.fusion = nn.Sequential(
        #     nn.Linear(2048, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 2048)
        # )        
        self.fusion = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048)
        )

    def forward(self, x):
        """MDGM Forward

        Args:
            x (Tensor): BxNxD

        Returns:
            Tensor: BxD
        """        
        B = x.shape[0]
        x = torch.cat([self.master_node.repeat(B, 1, 1), x], dim=1)        
        
        # Action Graph
        mixed_patterns = self.pattern_mixer(self.basic_patterns)
        
        x1 = self.mlp1(x)
        x1 = self.gcn1(x1, mixed_patterns)

        x1 = rearrange(x1, 'b h n d -> b n (h d)', h=self.num_head)
        x1 = F.normalize(x1[:, 0], dim=-1)

        # Content Graph
        x2 = self.mlp2(x)        
        x2 = self.gcn2(x2)

        x2 = rearrange(x2, 'b h n d -> b n (h d)', h=self.num_head)
        x2 = F.normalize(x2[:, 0], dim=-1)

        # Fusion
        # x = self.fusion(torch.cat([x1, x2], dim=-1))
        x = self.fusion(x1)
        x = F.normalize(x, dim=-1)

        return x


class V2G(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = Xception_Fea()

        self.roi = RoIAlign((1, 1), 1.0, -1, aligned=True)

        self.gcn = MultiDepGraphModule()

        self.pool = nn.MaxPool1d(8)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(1024, 2),
            nn.Softmax(1)
        )
    
    def forward(self, x, rois):
        """V2G forward

        Args:
            x (FloatTensor): BxCxHxW
            rois (FloatTensor): (B*5)*5, areas including whole face, left eye, right eye, nose and mouth

        Returns:
            FloatTensor: video-level predictions 
        """        
        B = x.shape[0]
        x = self.features(x)
        x0 = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        x0 = self.pool(x0.T).transpose(0, 1)
        # x0 = F.normalize(x0, dim=-1)

        x = self.roi(x, rois).flatten(1)        
        x = x.view(B//clip_len, clip_len*group_size, -1)
        x1 = self.gcn(x)
        x1 = F.normalize(x1, dim=-1)

        return self.classifier(x1)