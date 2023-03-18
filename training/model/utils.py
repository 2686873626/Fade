import torch

from torchvision.ops.misc import ConvNormActivation
from torch import Tensor, nn
from torch.nn import functional as F
from einops import rearrange


def get_sim_adj_mat(x):
    """get similarity adjacent matrix

    Args:
        x (FloatTensor): input feature map, BxNxD or BxNxCxHxW

    Returns:
        FloatTensor: adjacent matrix, BxNxN
    """    
    if len(x.shape)>3:
        x = x.clone().flatten(2) # avoid inplace operation

    # x.shape B N D
    mod = torch.norm(x, p=2, dim=-1)
    mod = torch.clamp(mod, 1e-8) # avoid 'DivZeroError'
    x = x/mod.unsqueeze(-1)
    mat = x@(x.transpose(1, 2))

    # B, N, _ = mat.shape
    # mask = 1. - torch.eye(N)    
    # mat = mat*(mask.to(x.device))

    return mat


def get_temp_adj_mat(vertex_num, group_size=5):
    """
    Args:
        vertex_num (int): noted as N
        group_size (int, optional): vertex number in one frame. Defaults to 5.

    Returns:
        FloatTensor: adjacent matrix, N*N
    """    
    # vertex_num=TxN
    # T-images sequence length
    # N-number of different features of one frame(all_frame, right eye, left eye, nose, mouth)
    mat = torch.zeros((vertex_num, vertex_num))
    for i in range(vertex_num):
        for j in range(vertex_num):
            if i==j:
                # self loop removed
                mat[i][j]=0.
            elif i//group_size==j//group_size:
                # same frame
                mat[i][j]=1.
            elif j//group_size-1==i//group_size:
                # next frame
                mat[i][j]=1.

    # remove the edges between eyes and mouths
    for i in range(vertex_num//group_size):
        for j in range(vertex_num//group_size):
            if i==j:
                mat[i*group_size+1][j*group_size+4]=0.
                mat[i*group_size+2][j*group_size+4]=0.
                mat[i*group_size+4][j*group_size+1]=0.
                mat[i*group_size+4][j*group_size+2]=0.
            elif j-i==1:
                mat[i*group_size+1][j*group_size+4]=0.
                mat[i*group_size+2][j*group_size+4]=0.
                mat[i*group_size+4][j*group_size+1]=0.
                mat[i*group_size+4][j*group_size+2]=0.
    
    # norm
    mat = mat/mat.sum(dim=1, keepdim=True)
    return mat


def get_basic_patterns(group_size=5, num_basic=2):
    """Set the basic action patterns

    Args:
        group_size (int, optional): the vertex count in one frame. Defaults to 5.

    Returns:
        Tensor: HxNxN
    """        
    template = torch.zeros(num_basic, group_size, group_size)
    if 5==group_size:
        # neutral face
        template[0] = torch.tensor([
            [1.0, 1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ])
        # simling face
        template[1] = torch.tensor([
            [1.0, 1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ])
        # serious face
        template[2] = torch.tensor([
            [1.0, 1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ])
        # positional relation
        template[3] = torch.tensor([
            [1.0, 1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ])
    if 7==group_size:
        # neural face
        template[0] = torch.tensor([
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ])
        # smiling face
        template[1] = torch.tensor([
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ])
        # serious face
        template[2] = torch.tensor([
            [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ])
        # anxious face
        template[3] = torch.tensor([
            [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ])
        # positional relation
        template[-2] = torch.tensor([
            [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ])

    template[-1] = torch.eye(group_size)

    return template


class SepConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        """Seperable Convolution

        Args:
            in_channels (int): _description_
            out_channels (int): _description_
            kernel_size (int, optional): _description_. Defaults to 3.
            stride (int, optional): _description_. Defaults to 1.
            padding (int, optional): _description_. Defaults to 0.
            dilation (int, optional): _description_. Defaults to 1.
            bias (bool, optional): _description_. Defaults to False.
        """        
        super().__init__()
        self.depthwise_conv = ConvNormActivation(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise_conv = ConvNormActivation(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias
        )
        
    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.dw_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(in_channels)
        self.pw_conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels*4
        )
        self.relu = nn.ReLU()
        self.pw_conv2 = nn.Conv2d(
            in_channels=in_channels*4,
            out_channels=out_channels
        )
    
    def forward(self, x):
        x = self.dw_conv(x)
        x = self.bn(x)
        x = self.pw_conv1(x)
        x = self.relu(x)
        x = self.pw_conv2(x)
        return x

