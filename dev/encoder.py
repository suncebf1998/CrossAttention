import torch
from torch import nn
import einops

class Head_Encode_1d(nn.Module):
    def __init__(self):
        super().__init__()
        

class CrossAttention1d(nn.Module):

    def __init__(
        self, head_encode:nn.Module, 
        generation:nn.Module, 
        proj:nn.ModuleList, 
        deproj:nn.ModuleList,
        num_heads=[3, 6, 12, 24],
        use_skip=True,
        drop_ratio=0.2):

        self.head_encode = head_encode
        self.gen_model = generation
        self.num_heads = num_heads
        self.encodelist = nn.ModuleList()
        self.projs = proj
        self.deprojs = deproj
        self.head_dim = []
        self.softmax = nn.Softmax(dim=-1)
        self.use_skip = use_skip
        self.dropout = nn.Dropout(drop_ratio) if drop_ratio>0 else nn.Identity()

    def forward(self, x, noise):
        x = self.head_encode # B, ... -> B, dim, num
        cross_outputs = reversed(self.gen_model(noise))
        for i, model, cross, proj, deproj in \
            enumerate(zip(self.encodelist, cross_outputs, self.projs, self.deprojs)):
            x_ori = model(x)
            x = einops.rearrange(
                x_ori, "B (dim numhead) num -> B dim (numhead num)", numhead=self.num_heads[i])
            cross_proj = proj(cross) # B, dimc, numc -> B, dim, numc
            cross_proj = einops.rearrange(
                cross_proj, "B (dim numhead) num -> B dim (numhead num)", numhead=self.num_heads[i])
            cross = einops.rearrange(
                cross, "B (dim numhead) num -> B (numhead num) dim", numhead=self.num_heads[i])
            atten = x.transpose(-1, -2) @ cross_proj * self.head_dim ** -0.5 # B, numx, numc
            x = atten @ cross # B numx dimc
            x = einops.rearrange(
                x, "B (num head_num) dim -> B num (head_num dim)", head_num=self.num_heads[i])
            x = deproj(x)
            if self.use_skip:
                x = x_ori + self.dropout(x)
        return x