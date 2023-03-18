import torch
import torch.nn.functional as F
from torchvision.ops import RoIAlign
from torch import nn, einsum
from einops import rearrange
from .utils import SepConv, get_sim_adj_mat, get_temp_adj_mat
from .features import Xception_Fea, EffNet_Fea, ResNet_Fea


clip_len = 8
group_size = 5


class ConvBasedGCN(nn.Module):
    def __init__(self, in_ch, out_ch, dynamic=False):
        super().__init__()
        self.layer1 = SepConv(in_ch, in_ch//2, padding=1, bias=False)
        self.layer2 = SepConv(in_ch//2, out_ch, padding=1, bias=False)
        self.dynamic = dynamic
        self.dropout = nn.Dropout2d()
        if dynamic:
            self.alpha1 = nn.Parameter(torch.tensor(1.))
            self.alpha2 = nn.Parameter(torch.tensor(1.))

    def forward(self, x, A=None):
        """
        Args:
            x (FloatTensor): BxNxCxHxW
            A (FloatTensor): adjacent matrix, BxNxN or NxN

        Returns:
            FloatTensor: x after message passing, BxNxCxHxW
        """        
        B = x.shape[0]
        if A is not None:
            A = A.unsqueeze(0).repeat(B, 1, 1)

        if self.dynamic:
            A = get_sim_adj_mat(x)        
            A = torch.softmax(A/self.alpha1.clamp(min=1e-2), -1)
            
        x = einsum('b m n, b n c h w -> b m c h w', A, x)
        x = rearrange(x, 'b m c h w -> (b m) c h w')
        x = self.dropout(x)
        x = self.layer1(x)
        x = rearrange(x, '(b n) c h w -> b n c h w', b=B)

        if self.dynamic:
            A = get_sim_adj_mat(x)        
            A = torch.softmax(A/self.alpha2.clamp(min=1e-2), -1)
            
        x = einsum('b m n, b n c h w -> b m c h w', A, x)
        x = rearrange(x, 'b m c h w -> (b m) c h w')
        x = self.dropout(x)
        x = self.layer2(x)
        x = rearrange(x, '(b m) c h w -> b m c h w', b=B)
        
        return x


class V2G(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = Xception_Fea()
        self.squeeze = nn.Conv2d(2048, 512, 1)
        # self.features = EffNet_Fea()
        # self.squeeze = nn.Conv2d(1792, 512, 1)
        # self.features = ResNet_Fea()
        # self.squeeze = nn.Conv2d(2048, 512, 1)

        self.roi = RoIAlign((3, 3), 1.0, -1, aligned=True)
        adj_mat = get_temp_adj_mat(clip_len*group_size, group_size)
        self.register_buffer('seq_mat', adj_mat)
        self.register_buffer('rev_mat', adj_mat.transpose(0, 1))

        self.gcn1 = ConvBasedGCN(512, 512)
        self.gcn2 = ConvBasedGCN(512, 512)
        self.gcn3 = ConvBasedGCN(512, 512, True)

        # self.pool = nn.AdaptiveMaxPool3d((1, 3, 3))
        self.pool = nn.AdaptiveAvgPool3d((1, 3, 3))

        self.fuse = nn.Sequential(
            nn.Conv2d(1024, 2048, 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(2048, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 2),
            nn.Softmax(1)
        )
        
    def forward(self, x, rois):
        B = x.shape[0]
        x0 = self.features(x)
        x0 = self.squeeze(x0)

        # partial features
        x = self.roi(x0, rois)
        x = x.view(B//clip_len, clip_len*5, 512, 3, 3)
        x1 = self.gcn1(x, self.seq_mat)
        x2 = self.gcn2(x, self.rev_mat)
        x3 = self.gcn3(x)
        x4 = x1+x2+x3+x
        
        # x0 = F.adaptive_avg_pool2d(F.relu(x0), (3, 3))
        x0 = F.adaptive_avg_pool2d(x0, (3, 3))
        x0 = x0.view(B//clip_len, clip_len, 512, 3, 3)
        
        # fusion
        x4.transpose_(1, 2)
        x0.transpose_(1, 2)
        x4 = self.pool(x4).squeeze()
        x0 = self.pool(x0).squeeze()
        if B==8:
            x4.unsqueeze_(0)
            x0.unsqueeze_(0)
        x = torch.cat([x4, x0], dim=1)
        x = self.fuse(x)

        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)

        return self.classifier(x)


if __name__ == '__main__':
    x = torch.randn(16, 3, 299, 299)
    rois = torch.randn(80, 5)
    for i in range(80):
        rois[i] = torch.tensor([i//5, 0., 0., 1., 1.])

    net = V2G()
    pred = net(x, rois)