from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math


@dataclass
class GPTConfig:
    block_size: int = 256  # max_len for input seq
    vocab_size: int = 65  # model vocab
    n_layer: int = 6  # model transformer blocks
    n_head: int = 6  # attention heads in each transformer block
    n_embed = 384  # model embedding size: token embed_size + positional embed_size


class Block(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x += self.attn(self.ln_1(x))
        x += self.mlp(self.ln_2(x))
        return x


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(
            approximate="tanh"
        )  # historically due to slow TF implementation
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

        def forward(self, x):
            B, T, C = x.size()

            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)

            y = att @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)

            y = self.proj(y)


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config.vocab_size, config.n_embed
                ),  # token embedding layer
                wpe=nn.Embedding(
                    config.block_size, config.n_embed
                ),  # positional embedding layer
                h=nn.ModuleList(
                    [Block(config) for _ in range(config.n_layer)]
                ),  # transformer blocks
                ln_f=nn.LayerNorm(
                    config.n_embed
                ),  # layer norm applied to the output of the last layer, addition to original transformers architecture
            )
        )
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
