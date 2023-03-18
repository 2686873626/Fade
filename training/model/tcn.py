import torch
from torch import nn
from pretrainedmodels import xception
from torchvision.models.video import r3d_18, mc3_18


clip_len = 8
group_size = 7

class V2G(nn.Module):
    def __init__(self):
        super().__init__()
        # self.features = mc3_18(pretrained=True)
        self.features = r3d_18(pretrained=True)
        for m in self.features.modules():
            if isinstance(m, nn.Conv3d):
                if m.stride == (2, 2, 2):
                    m.stride = (1, 2, 2)

        # self.features = InceptionI3d()
        # self.features.load_state_dict(torch.load('init_weights/i3d_init.pt'))
        self.features.fc = nn.Linear(512, 2)
        # self.features.replace_logits(num_classes=2)
    
    def forward(self, x, rois):
        """V2G forward

        Args:
            x (FloatTensor): BxCxHxW
            rois (FloatTensor): (B*7)*5, areas including whole face, left eye, right eye, nose and mouth

        Returns:
            FloatTensor: video-level predictions 
        """      
        B, C, H, W = x.shape
        x = x.view(B//clip_len, clip_len, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        # x = self.features.extract_features(x)
        x = self.features(x)

        return torch.softmax(x, -1)