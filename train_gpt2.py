from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


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
                ),  # layer norm applied to the output of the last layer
            )
        )
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
