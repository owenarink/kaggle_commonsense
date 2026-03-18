import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def build_relative_positions(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Returns relative position matrix of shape [L, L]:
    entry (i, j) = j - i
    """
    position_ids = torch.arange(seq_len, device=device)
    rel_pos = position_ids[None, :] - position_ids[:, None]
    return rel_pos


class DisentangledMultiheadAttention(nn.Module):
    """
    Simplified DeBERTa-style disentangled self-attention.

    Attention score = content->content + content->position + position->content

    - content->content: token query attends to token key
    - content->position: token query attends to relative position key
    - position->content: relative position query attends to token key

    This is a simplified version inspired by the official DeBERTa implementation,
    which separately projects content and position interactions inside attention. :contentReference[oaicite:1]{index=1}
    """
    def __init__(self, input_dim, embed_dim, num_heads, max_len=512, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_len = max_len
        self.max_relative_positions = max_len

        # Content projections
        self.q_proj = nn.Linear(input_dim, embed_dim)
        self.k_proj = nn.Linear(input_dim, embed_dim)
        self.v_proj = nn.Linear(input_dim, embed_dim)

        # Relative position embeddings
        self.rel_pos_emb = nn.Embedding(2 * max_len - 1, input_dim)

        # Position projections
        self.pos_key_proj = nn.Linear(input_dim, embed_dim)
        self.pos_query_proj = nn.Linear(input_dim, embed_dim)

        self.o_proj = nn.Linear(embed_dim, input_dim)
        self.attn_drop = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.pos_key_proj.weight)
        nn.init.xavier_uniform_(self.pos_query_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

        self.q_proj.bias.data.fill_(0)
        self.k_proj.bias.data.fill_(0)
        self.v_proj.bias.data.fill_(0)
        self.pos_key_proj.bias.data.fill_(0)
        self.pos_query_proj.bias.data.fill_(0)
        self.o_proj.bias.data.fill_(0)

    def _shape(self, x, batch_size, seq_length):
        # [B, L, E] -> [B, H, L, D]
        return x.view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _get_rel_pos_embeddings(self, seq_length, device):
        """
        Build relative position embeddings for all pairwise offsets.
        Output: [L, L, input_dim]
        """
        rel_pos = build_relative_positions(seq_length, device=device)
        rel_pos = rel_pos.clamp(
            min=-(self.max_relative_positions - 1),
            max=(self.max_relative_positions - 1),
        )
        rel_pos = rel_pos + (self.max_relative_positions - 1)
        rel_embeddings = self.rel_pos_emb(rel_pos)  # [L, L, input_dim]
        return rel_embeddings

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()

        if mask is not None:
            mask = expand_mask(mask)  # [B, 1, 1, L] or compatible

        # Content projections
        q = self._shape(self.q_proj(x), batch_size, seq_length)  # [B, H, L, D]
        k = self._shape(self.k_proj(x), batch_size, seq_length)  # [B, H, L, D]
        v = self._shape(self.v_proj(x), batch_size, seq_length)  # [B, H, L, D]

        # Relative position embeddings + projections
        rel_embeddings = self._get_rel_pos_embeddings(seq_length, x.device)  # [L, L, input_dim]
        pos_k = self.pos_key_proj(rel_embeddings)    # [L, L, E]
        pos_q = self.pos_query_proj(rel_embeddings)  # [L, L, E]

        pos_k = pos_k.view(seq_length, seq_length, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
        pos_q = pos_q.view(seq_length, seq_length, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
        # pos_k, pos_q: [H, L, L, D]

        # 1) content-to-content: q_i · k_j
        c2c = torch.matmul(q, k.transpose(-2, -1))  # [B, H, L, L]

        # 2) content-to-position: q_i · r_{i,j}^{key}
        c2p = torch.einsum("bhid,hijd->bhij", q, pos_k)  # [B, H, L, L]

        # 3) position-to-content: r_{i,j}^{query} · k_j
        p2c = torch.einsum("hijd,bhjd->bhij", pos_q, k)  # [B, H, L, L]

        # DeBERTa scales attention using the number of active score terms;
        # the official implementation adjusts scale_factor depending on which
        # disentangled terms are included. :contentReference[oaicite:2]{index=2}
        scale_factor = 3.0
        attn_logits = (c2c + c2p + p2c) / math.sqrt(self.head_dim * scale_factor)

        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)

        attention = F.softmax(attn_logits, dim=-1)
        attention = self.attn_drop(attention)

        values = torch.matmul(attention, v)  # [B, H, L, D]
        values = values.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        return o


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, dim_feedforward, num_heads, max_len=512, dropout=0.0, act_fn=nn.ReLU):
        super().__init__()
        self.self_attn = DisentangledMultiheadAttention(
            input_dim=input_dim,
            embed_dim=input_dim,
            num_heads=num_heads,
            max_len=max_len,
            dropout=dropout,
        )

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
        for i in self.layers:
            x = i(x, mask=mask)
        return x

    @torch.no_grad()
    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for i in self.layers:
            attn_out, attn_map = i.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = i.norm1(x + i.drop(attn_out))
            ff_out = i.ff(x)
            x = i.norm2(x + i.drop(ff_out))
        return attention_maps


class PositionalEncoding(nn.Module):
    """
    Kept here in case you still want absolute position information added to inputs.
    DeBERTa mainly uses disentangled relative position interactions inside attention,
    rather than relying only on summed input position embeddings. :contentReference[oaicite:3]{index=3}
    """
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


class BertTransformer(nn.Module):
    """
    For your problem:
    - input: token ids [B, L]
    - output:
        num_classes=3 => logits [B, 3]
        num_classes=1 => logits [B, 1]

    This version uses disentangled attention:
    - content-to-content
    - content-to-position
    - position-to-content
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
        pooling="mean",   # "mean" or "cls"
        use_absolute_pos=False,
    ):
        super().__init__()
        assert pooling in ("mean", "cls")

        self.pad_idx = pad_idx
        self.pooling = pooling
        self.use_absolute_pos = use_absolute_pos

        self.embedding = nn.Embedding(vocab_size, model_dim, padding_idx=pad_idx)

        # Optional absolute PE, turned off by default because disentangled attention
        # already injects relative positional information inside attention.
        self.positional_encoding = PositionalEncoding(d_model=model_dim, max_len=max_len)

        self.transformer = TransformerEncoder(
            num_layers=num_layers,
            input_dim=model_dim,
            dim_feedforward=ff_mult * model_dim,
            num_heads=num_heads,
            max_len=max_len,
            dropout=dropout,
            act_fn=nn.ReLU,
        )

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(model_dim, num_classes)

    def forward(self, x_ids):
        pad_mask = (x_ids != self.pad_idx)  # [B, L] bool

        x = self.embedding(x_ids)  # [B, L, D]

        if self.use_absolute_pos:
            x = self.positional_encoding(x)

        x = self.transformer(x, mask=pad_mask)

        if self.pooling == "mean":
            m = pad_mask.unsqueeze(-1).float()      # [B, L, 1]
            denom = m.sum(dim=1).clamp(min=1.0)     # [B, 1]
            x = (x * m).sum(dim=1) / denom          # [B, D]
        else:
            x = x[:, 0, :]

        x = self.dropout(x)
        logits = self.out(x)
        return logits

    @torch.no_grad()
    def get_attention_maps(self, x_ids):
        pad_mask = (x_ids != self.pad_idx).to(x_ids.device)
        x = self.embedding(x_ids)
        if self.use_absolute_pos:
            x = self.positional_encoding(x)
        return self.transformer.get_attention_maps(x, mask=pad_mask)


if __name__ == "__main__":
    B, L = 2, 8
    vocab_size = 100
    x = torch.randint(0, vocab_size, (B, L))
    x[:, -2:] = 0

    clf = BertTransformer(
        vocab_size=vocab_size,
        num_classes=1,
        pad_idx=0,
        model_dim=64,
        num_heads=4,
        num_layers=2,
        max_len=128,
        pooling="mean",
        use_absolute_pos=False,
    )
    y = clf(x)
    print("BertTransformer logits:", y.shape)  # [B, 1]
