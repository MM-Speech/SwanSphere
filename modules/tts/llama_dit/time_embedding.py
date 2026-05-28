import math
import torch
from math import pi
from einops import rearrange, reduce
from torch import Tensor, nn
from typing import Sequence, Union
from torch.nn import functional as F
from torch.distributions import LogisticNormal


class NumberEmbedder(nn.Module):
    def __init__(self, features: int, dim: int = 256):
        super().__init__()
        assert dim % 2 == 0, f"dim must be divisible by 2, found {dim}"
        self.features = features
        self.weights = nn.Parameter(torch.randn(dim // 2))
        self.to_out = nn.Linear(in_features=dim + 1, out_features=features)

    def to_embedding(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return self.to_out(fouriered)

    def forward(self, x: Union[Sequence[float], Tensor]) -> Tensor:
        if not torch.is_tensor(x):
            x = torch.tensor(x, device=self.weights.device)
        assert isinstance(x, Tensor)
        shape = x.shape
        x = rearrange(x, "... -> (...)")
        return self.to_embedding(x).view(*shape, self.features)  # type: ignore


class DDPMTimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.hidden_size = hidden_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        t = ((1-t)*1000).int()
        print(t)
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.hidden_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    

def Ts(t):
    """Builds a type template for a given type that accepts a list of instances"""
    return lambda *types: lambda: t(*[tp() for tp in types])


class Sequential(nn.Module):
    """Custom Sequential that includes all args"""

    def __init__(self, *blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor, *args) -> Tensor:
        for block in self.blocks:
            x = block(x, *args)
        return x


def Repeat(m, times: int):
    ms = (m,) * times
    return Sequential(*ms) if isinstance(m, nn.Module) else Ts(Sequential)(*ms)


class CFMTimeEmbedding(nn.Module):
    def __init__(self, modulation_features, num_layers: int = 2, bias=True):
        super().__init__()
        self.embedding = NumberEmbedder(features=modulation_features)
        self.mlp = Repeat(
            nn.Sequential(
                nn.Linear(modulation_features, modulation_features, bias=bias),
                nn.GELU(),
            ),
            times=num_layers,
        )

    def forward(self, time):
        # Process time to time_features
        time_features = F.gelu(self.embedding(time))
        time_features = self.mlp(time_features)
        # Overlap features if more than one per batch
        if time_features.ndim == 3:
            time_features = reduce(time_features, "b n d -> b d", "sum")

        return time_features


class LogitNormalTrainingTimesteps:
    def __init__(self, T=1000.0, loc=0.0, scale=1.0):
        assert T > 0
        self.T = T
        self.dist = LogisticNormal(loc, scale)

    def sample(self, size, device):
        t = self.dist.sample(size)[..., 0].to(device)
        return t


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch

    sampler = LogitNormalTrainingTimesteps()
    sampled_data = sampler.sample([1], 'cpu')
    print(sampled_data)