import torch
from torch import nn
from torch.nn import Linear
from modules.tts.ar_dur.commons.layers import LayerNorm, Embedding


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class ConvBlock(nn.Module):
    def __init__(self, idim=80, n_chans=256, kernel_size=3, stride=1, norm='gn', dropout=0):
        super().__init__()
        self.conv = ConvNorm(idim, n_chans, kernel_size, stride=stride)
        self.norm = norm
        if self.norm == 'bn':
            self.norm = nn.BatchNorm1d(n_chans)
        elif self.norm == 'in':
            self.norm = nn.InstanceNorm1d(n_chans, affine=True)
        elif self.norm == 'gn':
            self.norm = nn.GroupNorm(n_chans // 16, n_chans)
        elif self.norm == 'ln':
            self.norm = LayerNorm(n_chans // 16, n_chans)
        elif self.norm == 'wn':
            self.conv = torch.nn.utils.weight_norm(self.conv.conv)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        """

        :param x: [B, C, T]
        :return: [B, C, T]
        """
        x = self.conv(x)
        if not isinstance(self.norm, str):
            if self.norm == 'none':
                pass
            elif self.norm == 'ln':
                x = self.norm(x.transpose(1, 2)).transpose(1, 2)
            else:
                x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ConvGlobalStacks(nn.Module):
    def __init__(self, idim=80, n_layers=5, n_chans=256, odim=32, kernel_size=5, norm='gn', dropout=0,
                 strides=[2, 2, 2, 2, 2]):
        super().__init__()
        self.conv = torch.nn.ModuleList()
        self.pooling = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        self.in_proj = Linear(idim, n_chans)
        for idx in range(n_layers):
            self.conv.append(ConvBlock(n_chans, n_chans, kernel_size, stride=strides[idx],
                                       norm=norm, dropout=dropout))
        self.out_proj = Linear(n_chans, odim)

    def forward(self, x):
        """

        :param x: [B, T, H]
        :return: [B, T, H]
        """
        x_nonpadding = (x.abs().sum(-1) > 0).float()
        x = self.in_proj(x) * x_nonpadding[..., None]
        x_nonpadding = x_nonpadding[:, None]
        x = x.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            x_nonpadding = x_nonpadding[:, :, ::2]
            x = f(x)  # (B, C, T)
            x = x * x_nonpadding
        x = x.transpose(1, -1)
        x = x.sum(1) / x_nonpadding.sum(-1)
        x = self.out_proj(x)  # (B, H)
        return x