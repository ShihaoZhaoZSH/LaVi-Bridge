from dataclasses import dataclass
from inspect import isfunction

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.utils import BaseOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


def default(val, d):
    if val is not None: return val
    return d() if isfunction(d) else d


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out, mult=4, dropout=0.1):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = GEGLU(dim, inner_dim)
        self.net = nn.Sequential(
            project_in, 
            nn.Dropout(dropout), 
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)
    

@dataclass
class TextAdapterOutput(BaseOutput):
    sample: torch.FloatTensor


class TextAdapter(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, in_dim, int_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.ff1 = FeedForward(in_dim, int_dim)
        self.ff2 = FeedForward(int_dim, out_dim)
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(int_dim)

    def forward(self, x):
        x = self.ff1(self.norm1(x))
        x = self.ff2(self.norm2(x))
        return TextAdapterOutput(x)