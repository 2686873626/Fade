import torch
import torch.nn.functional as F

from torchvision.ops import RoIAlign
from torch import nn
from .utils import get_sim_adj_mat, get_temp_adj_mat
from .features import Xception_Fea


class GCN(nn.Module):
    def __init__(self, in_ch, out_ch, dynamic=False):
        super().__init__()
        self.layer1 = nn.Linear(in_ch, in_ch//2)
        self.layer2 = nn.Linear(in_ch//2, out_ch)

        self.norm1 = nn.LayerNorm(in_ch)
        self.norm2 = nn.LayerNorm(in_ch//2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

        self.dynamic = dynamic
        
    def forward(self, x, A=None):
        """gcn forward

        Args:
            x (FloatTensor): BxNxD
            A (FloatTensor): BxNxN or NxN

        Returns:
            FloatTensor: x after massage passing
        """
        x = self.norm1(x)
        x = self.dropout(x)
        if self.dynamic:
            mat = get_sim_adj_mat(x)
            A = torch.softmax(mat, -1)

        x = self.relu(self.layer1(A@x))
        x = self.norm2(x)
        x = self.dropout(x)
        if self.dynamic:
            mat = get_sim_adj_mat(x)
            A = torch.softmax(mat, -1)
        
        x = self.relu(self.layer2(A@x))
        return x


class V2G(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = Xception_Fea()
        self.squeeze = nn.Conv2d(2048, 512, 1)

        self.roi = RoIAlign((1, 1), 1.0, -1, aligned=True)
        adj_mat = get_temp_adj_mat(40, 5)
        self.forward_mat = nn.Parameter(adj_mat, requires_grad=False)
        self.backward_mat = nn.Parameter(adj_mat.transpose(0, 1), requires_grad=False)
        self.gcn1 = GCN(512, 512)
        self.gcn2 = GCN(512, 512)
        self.gcn3 = GCN(512, 512, True)
        # self.merge_1 = nn.Conv1d(8, 1, 1)
        self.merge_1 = nn.AdaptiveMaxPool2d((1, 512))
        # self.merge_2 = nn.Conv1d(40, 1, 1)
        self.merge_2 = nn.AdaptiveMaxPool2d((1, 512))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 2),
            # nn.Linear(2048, 2),
            nn.Softmax(1)
        )
    
    def forward(self, x, rois):
        """V2G forward

        Args:
            x (FloatTensor): (B * T) * C * H * W
            rois (FloatTensor): (B * T * count_areas) * 5, areas including whole face, left eye, right eye, left_cheek, right_cheek, nose and mouth
                                [frame idx, x1, y1, x2, y2]

        Returns:
            FloatTensor: video-level predictions 
        """        
        B = x.shape[0]
        x = self.features(x)

        x = self.squeeze(x)

        x0 = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        x = self.roi(x, rois).flatten(1)
        
        x = x.view(B // 8, 40, -1)
        x1 = self.gcn1(x, self.forward_mat)
        x2 = self.gcn2(x, self.backward_mat)
        x3 = self.gcn3(x)
        graph_feas = x1+x2+x3+x
        graph_feas = self.merge_2(graph_feas).squeeze()

        img_feas = x0.view(B//8, 8, -1)
        img_feas = self.merge_1(img_feas).squeeze()
        graph_feas = F.normalize(graph_feas)
        img_feas = F.normalize(img_feas)
        preds = self.classifier(torch.cat([img_feas, graph_feas], dim=1))
        
        return preds
