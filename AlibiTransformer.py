import math
import torch
from torch import nn
from torch.nn import functional as F

def get_relative_positions(seq_len: int) -> torch.tensor:
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    return x - y


def get_alibi_slope(num_heads):
    x = (2 ** 8) ** (1 / num_heads)
    return (
        torch.tensor([1 / x ** (i + 1) for i in range(num_heads)])
        .unsqueeze(-1)
        .unsqueeze(-1)
    )


class ALiBiMultiHeadAttention(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.causal = kwargs["causal"]
        self.num_heads = kwargs["n_head"]
        self.scale = math.sqrt(kwargs["n_embd"])
        self.dropout = nn.Dropout(kwargs["dropout"])
        self.register_buffer("m", get_alibi_slope(self.num_heads))
        self.kqv = nn.Linear(kwargs["n_embd"], 3 * kwargs["n_embd"], bias=False)
        if kwargs["causal"]:
            self.register_buffer(
                "mask", torch.tril(torch.ones(1, 1, kwargs["block_size"], kwargs["block_size"]))
            )
        self.device = kwargs["device"]

    def forward(self, x: torch.tensor) -> torch.tensor:
        batch_size, seq_len, _ = x.shape

        key, query, value = self.kqv(x).chunk(3, dim=-1)
        key = key.view(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # key.shape == (batch_size, num_heads, d_head, seq_len)
        query = query.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # qv.shape == (batch_size, num_heads, seq_len, d_head)

        relative_positions=torch.tensor(get_relative_positions(seq_len)).to(self.device)
        bias = (self.m * relative_positions).unsqueeze(0)
        # bias.shape == (1, num_heads, seq_len, seq_len)

        score = torch.matmul(query, key) / self.scale + bias
        # score.shape == (batch_size, num_heads, seq_len, seq_len)

        if self.causal:
            score = score.masked_fill(
                self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf")
            )

        attn = F.softmax(score, dim=-1)
        out = torch.matmul(attn, value)
        # out.shape == (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, n_embd)
        out = self.dropout(out)

        return out, attn

class FeedForward(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.fc1 = nn.Linear(kwargs["n_embd"], kwargs["n_hidden"])
        self.fc2 = nn.Linear(kwargs["n_hidden"], kwargs["n_embd"])
        self.dropout = nn.Dropout(kwargs["dropout"])

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = F.gelu(self.fc1(x))
        out = self.dropout(self.fc2(x))
        return out


class ALiBiTransformerLayer(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.ffn_norm = nn.LayerNorm(kwargs["n_embd"])
        self.attn_norm = nn.LayerNorm(kwargs["n_embd"])
        self.ffn = FeedForward(**kwargs)
        self.attn = ALiBiMultiHeadAttention(**kwargs)

    def forward(self, x: torch.tensor) -> torch.tensor:
        out, attn = self.attn(self.attn_norm(x))
        self.attn_map = attn
        x = x + out
        x = x + self.ffn(self.ffn_norm(x))
        return x
    
class ALiBiTransformer(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(kwargs["vocab_size"], kwargs["n_embd"])
        self.ln_f = nn.LayerNorm(kwargs["n_embd"]) # final layer norm
        self.lm_head = nn.Linear(kwargs["n_embd"], kwargs["vocab_size"])
        self.block_size = kwargs["block_size"]
        self.device = kwargs["device"]
        self.layers = nn.Sequential(
            *[ALiBiTransformerLayer(**kwargs) for _ in range(kwargs["n_layer"])]
        )

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        x = self.layers(tok_emb)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        attn_map = self.layers[-1].attn_map
        return logits, loss, attn_map