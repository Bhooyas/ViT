import torch
from torch import nn
import math
from einops.layers.torch import Rearrange

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, heads, d_head, dropout):
        super(MultiHeadAttention, self).__init__()

        head_dim = d_head * heads
        self.heads = heads
        self.scale = head_dim ** -0.5

        self.norm = nn.LayerNorm(d_model)
        self.act = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.qkv = nn.Linear(d_model, head_dim * 3, bias=False)
        self.rearrange = Rearrange("b n (h d) -> b h n d", h=self.heads)
        self.rearrange_out = Rearrange("b h n d -> b n (h d)", h=self.heads)
        self.out = nn.Linear(head_dim, d_model, bias=False)

    def forward(self, x):
        x = self.norm(x)

        Q, K, V = self.qkv(x).chunk(3, dim=-1)

        Q = self.rearrange(Q)
        K = self.rearrange(K)
        V = self.rearrange(V)

        attn = self.act(Q @ K.transpose(2, 3) * self.scale)

        y = attn @ V
        y = self.rearrange_out(y)
        return self.out(y)

class FeedForward(nn.Module):

    def __init__(self, d_model, factor, dropout):
        super(FeedForward, self).__init__()

        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * factor),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(d_model * factor, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.fc(x)

class Encoder(nn.Module):

    def __init__(self, d_model, heads, d_head, dropout, factor):
        super(Encoder, self).__init__()

        self.encoder = nn.ModuleList([
            MultiHeadAttention(d_model, heads, d_head, dropout),
            FeedForward(d_model, factor, dropout)
        ])

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x) + x
        return x

class ViT(nn.Module):

    def __init__(self, image_size, patch_size, num_classes, depth, d_model, heads, dropout, d_head=64, factor=4, channel=3, temperature=10000):
        super(ViT, self).__init__()

        self.image_size = image_size
        self.patch_size = patch_size

        assert image_size % patch_size == 0, f"{image_size} shoud be divisible by {patch_size}"

        self.patch_dim = (image_size // patch_size) ** 2 * channel
        self.num_patchs = (image_size // patch_size) ** 2

        self.patch_encoder = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=patch_size, p2=patch_size),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, d_model),
            nn.LayerNorm(d_model)
        )

        num_patch = (image_size // patch_size) * 2
        self.pe = nn.Embedding(self.num_patchs, d_model)
        self.register_buffer("pe_input", torch.arange(self.num_patchs).expand((1, -1)))


        self.encoders = nn.ModuleList([
            Encoder(d_model, heads, d_head, dropout, factor)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.to_class = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.patch_encoder(x)
        x = x + self.pe(self.pe_input)
        for encoder in self.encoders:
            x = encoder(x)
        x = self.norm(x)
        x = self.dropout(x.mean(dim=1))
        x = self.to_class(x)
        return x


if __name__ == '__main__':
    vit = ViT(image_size=32, patch_size=4, num_classes=10, depth=6, d_model=384, heads=6, factor=4)
    y = vit(torch.randn(1, 3, 32, 32))
    print(y.shape)
    print(vit)
