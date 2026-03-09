import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


def expand_mask(mask: torch.Tensor) -> torch.Tensor:
    # Supports:
    #  - [B, L] padding mask  -> [B, 1, 1, L]
    #  - [B, L, L]            -> [B, 1, L, L]
    #  - [B, H, L, L]         -> as is
    #  - [L, L]               -> [1, 1, L, L]
    assert mask.ndim >= 2, "Mask must be at least 2D"
    if mask.ndim == 2:              # [B, L]
        mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
    elif mask.ndim == 3:            # [B, L, L]
        mask = mask.unsqueeze(1)    # [B, 1, L, L]
    elif mask.ndim == 4:
        pass
    else:
        while mask.ndim < 4:
            mask = mask.unsqueeze(0)
    return mask


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, input_dim)
        self.attn_drop = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        if mask is not None:
            mask = expand_mask(mask)

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [B, H, L, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)

        values, attention = scaled_dot_product(q, k, v, mask=mask)
        attention = self.attn_drop(attention)

        values = values.permute(0, 2, 1, 3)  # [B, L, H, D]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        return o


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, dim_feedforward, num_heads, dropout=0.0, act_fn=nn.ReLU):
        super().__init__()
        self.self_attn = MultiheadAttention(input_dim=input_dim, embed_dim=input_dim, num_heads=num_heads, dropout=dropout)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        self.drop = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            act_fn(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, input_dim),
        )

    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, mask=mask)
        x = self.norm1(x + self.drop(attn_out))

        ff_out = self.ff(x)
        x = self.norm2(x + self.drop(ff_out))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x

    @torch.no_grad()
    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x, mask=mask)
        return attention_maps


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TextTransformer(nn.Module):
    """
    For your problem:
    - input: token ids [B, L] from preprocess_cnn(..., mode="multiclass"/"pairwise")
    - output:
        num_classes=3 => logits [B, 3] (CrossEntropyLoss)
        num_classes=1 => logits [B, 1] (BCEWithLogitsLoss) for pairwise scoring
    """
    def __init__(
        self,
        vocab_size,
        num_classes=3,
        pad_idx=0,
        model_dim=256,
        num_heads=8,
        num_layers=4,
        ff_mult=2,
        dropout=0.1,
        max_len=512,
        pooling="mean",  # "mean" or "cls"
    ):
        super().__init__()
        assert pooling in ("mean", "cls")

        self.pad_idx = pad_idx
        self.pooling = pooling

        self.embedding = nn.Embedding(vocab_size, model_dim, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(d_model=model_dim, max_len=max_len)

        self.transformer = TransformerEncoder(
            num_layers=num_layers,
            input_dim=model_dim,
            dim_feedforward=ff_mult * model_dim,
            num_heads=num_heads,
            dropout=dropout,
            act_fn=nn.ReLU,
        )

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(model_dim, num_classes)

    def forward(self, x_ids):
        # x_ids: [B, L] long
        pad_mask = (x_ids != self.pad_idx).to(x_ids.device)  # [B, L] with 1 for real tokens

        x = self.embedding(x_ids)                # [B, L, D]
        x = self.positional_encoding(x)          # [B, L, D]
        x = self.transformer(x, mask=pad_mask)   # mask will be expanded to [B,1,1,L]

        if self.pooling == "cls":
            pooled = x[:, 0, :]                  # [B, D]
        else:
            denom = pad_mask.sum(dim=1, keepdim=True).clamp_min(1)      # [B,1]
            pooled = (x * pad_mask.unsqueeze(-1)).sum(dim=1) / denom    # [B, D]

        pooled = self.dropout(pooled)
        logits = self.out(pooled)                # [B, C]
        return logits

    @torch.no_grad()
    def get_attention_maps(self, x_ids):
        pad_mask = (x_ids != self.pad_idx).to(x_ids.device)
        x = self.embedding(x_ids)
        x = self.positional_encoding(x)
        return self.transformer.get_attention_maps(x, mask=pad_mask)


if __name__ == "__main__":
    # quick sanity
    B, L = 2, 8
    vocab_size = 100
    x = torch.randint(0, vocab_size, (B, L))
    x[:, -2:] = 0  # pretend padding
    model = TextTransformer(vocab_size=vocab_size, num_classes=1, pad_idx=0, model_dim=64, num_heads=4, num_layers=2, max_len=128)
    y = model(x)
    print("logits shape:", y.shape)  # [B, 1]
