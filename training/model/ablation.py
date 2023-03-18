import torch
import torch.nn.functional as F
from torchvision.ops import RoIAlign
from torch import nn, einsum
from einops import rearrange
from .utils import SepConv, Block, get_sim_adj_mat, get_temp_adj_mat
from .features import Xception_Fea, EffNet_Fea


clip_len = 8


# class ConvBasedGCN(nn.Module):
#     def __init__(self, in_ch, out_ch, dynamic=False):
#         super().__init__()
#         self.layer1 = SepConv(in_ch, in_ch//2, padding=1, bias=True)
#         self.layer2 = SepConv(in_ch//2, out_ch, padding=1, bias=True)
#         self.dynamic = dynamic

#     def forward(self, x, A=None):
#         """
#         Args:
#             x (FloatTensor): BxNxCxHxW
#             A (FloatTensor): adjacent matrix, BxNxN or NxN

#         Returns:
#             FloatTensor: x after message passing, BxNxCxHxW
#         """        
#         B = x.shape[0]
#         if A is not None:
#             A = A.unsqueeze(0).repeat(B, 1, 1)

#         if self.dynamic:
#             mat = get_sim_adj_mat(x)
#             A = torch.softmax(mat, -1)
            
#         x = einsum('b m n, b n c h w -> b m c h w', A, x)
#         x = rearrange(x, 'b m c h w -> (b m) c h w')
#         x = F.dropout2d(x)
#         x = self.layer1(x)
#         x = rearrange(x, '(b n) c h w -> b n c h w', b=B)

#         if self.dynamic:
#             mat = get_sim_adj_mat(x)
#             A = torch.softmax(mat, -1)
            
#         x = einsum('b m n, b n c h w -> b m c h w', A, x)
#         x = rearrange(x, 'b m c h w -> (b m) c h w')
#         x = F.dropout2d(x)
#         x = self.layer2(x)
#         x = rearrange(x, '(b m) c h w -> b m c h w', b=B)
        
#         return x


# class ConvBasedGCN(nn.Module):
#     def __init__(self, in_ch, out_ch, dynamic=False):
#         super().__init__()
#         self.layer1 = SepConv(in_ch, in_ch, padding=1, bias=True)
#         self.dynamic = dynamic
#         self.iter = 3

#     def forward(self, x, A=None):
#         """
#         Args:
#             x (FloatTensor): BxNxCxHxW
#             A (FloatTensor): adjacent matrix, BxNxN or NxN

#         Returns:
#             FloatTensor: x after message passing, BxNxCxHxW
#         """        
#         B = x.shape[0]
#         if A is not None:
#             A = A.unsqueeze(0).repeat(B, 1, 1)

#         for _ in range(self.iter):

#             if self.dynamic:
#                 mat = get_sim_adj_mat(x)
#                 A = torch.softmax(mat, -1)
        
#             x = einsum('b m n, b n c h w -> b m c h w', A, x)
#             x = rearrange(x, 'b m c h w -> (b m) c h w')
#             x = F.dropout2d(x)
#             x = self.layer1(x)
#             x = rearrange(x, '(b n) c h w -> b n c h w', b=B)

#         return x


class ConvBasedGCN(nn.Module):
    def __init__(self, in_ch, out_ch, dynamic=False):
        super().__init__()
        self.layer1 = SepConv(in_ch, in_ch//2, padding=1, bias=True)
        self.layer2 = SepConv(in_ch//2, out_ch, padding=1, bias=True)
        self.dynamic = dynamic
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
            A = torch.softmax(A/self.alpha1.clamp(min=0.01), -1)
            
        x = einsum('b m n, b n c h w -> b m c h w', A, x)
        x = rearrange(x, 'b m c h w -> (b m) c h w')
        x = F.dropout2d(x)
        x = self.layer1(x)
        x = rearrange(x, '(b n) c h w -> b n c h w', b=B)

        if self.dynamic:
            A = get_sim_adj_mat(x)        
            A = torch.softmax(A/self.alpha2.clamp(min=0.01), -1)
            
        x = einsum('b m n, b n c h w -> b m c h w', A, x)
        x = rearrange(x, 'b m c h w -> (b m) c h w')
        x = F.dropout2d(x)
        x = self.layer2(x)
        x = rearrange(x, '(b m) c h w -> b m c h w', b=B)
        
        return x


class V2G(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = Xception_Fea()
        # self.features = EffNet_Fea()

        self.roi = RoIAlign((1, 1), 1.0, -1, aligned=True)
        adj_mat = get_temp_adj_mat(clip_len*5, 5)
        self.register_buffer('forward_mat', adj_mat)
        self.register_buffer('backward_mat', adj_mat.transpose(0, 1))
        # self.forward_mat = nn.Parameter(adj_mat, requires_grad=False)
        # self.backward_mat = nn.Parameter(adj_mat.transpose(0, 1), requires_grad=False)

        self.squeeze = nn.Conv2d(2048, 512, 1)
        # self.squeeze = nn.Conv2d(1792, 512, 1)

        # self.gcn1 = ConvBasedGCN(512, 512)
        # self.gcn2 = ConvBasedGCN(512, 512)
        # self.gcn3 = ConvBasedGCN(512, 512, True)

        self.pool = nn.AdaptiveMaxPool3d((1, 1, 1))

        # self.fuse = nn.Sequential(
        #     nn.Conv2d(1024, 2048, 1),
        #     nn.BatchNorm2d(2048),
        #     nn.ReLU(),
        #     nn.Dropout2d(0.3), # improve a lot
        #     nn.Conv2d(2048, 1024, 1),
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU()
        # )
        # self.fuse = nn.Sequential(
        #     nn.Linear(1024, 2048),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(2048, 1024),
        #     nn.ReLU()
        # )

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
        x4 = x.view(B//clip_len, clip_len*5, 512, 1, 1)
        # x1 = self.gcn1(x, self.forward_mat)
        # x2 = self.gcn2(x, self.backward_mat)
        # x3 = self.gcn3(x)
        # x4 = x1+x2+x3+x
        
        x0 = F.adaptive_avg_pool2d(x0, (1, 1))
        x0 = x0.view(B//clip_len, clip_len, 512, 1, 1)
        
        # fusion
        x4.transpose_(1, 2)
        x0.transpose_(1, 2)
        x4 = self.pool(x4).squeeze()
        x0 = self.pool(x0).squeeze()
        x = torch.cat([x4, x0], dim=1)
        # x = self.fuse(x)

        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)

        return self.classifier(x)


if __name__ == '__main__':
    x = torch.randn(16, 3, 299, 299)
    rois = torch.randn(80, 5)
    for i in range(80):
        rois[i] = torch.tensor([i//5, 0., 0., 1., 1.])

    net = V2G()
    pred = net(x, rois)