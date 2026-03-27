import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.bert import expand_mask, build_relative_positions, PositionalEncoding


def first_sep_index(x_ids: torch.Tensor, sep_idx: int, pad_idx: int) -> torch.Tensor:
    B, L = x_ids.shape
    sep_mask = (x_ids == sep_idx)
    has_sep = sep_mask.any(dim=1)
    idxs = torch.argmax(sep_mask.int(), dim=1)
    valid_len = (x_ids != pad_idx).sum(dim=1)
    fallback = torch.clamp(valid_len // 2, min=1, max=max(1, L - 2))
    return torch.where(has_sep, idxs, fallback)


def build_sep_distances(x_ids: torch.Tensor, pad_idx: int, sep_idx: int = None):
    B, L = x_ids.shape
    valid_mask = (x_ids != pad_idx)
    positions = torch.arange(L, device=x_ids.device).unsqueeze(0).expand(B, L)
    if sep_idx is not None:
        sep_pos = first_sep_index(x_ids, sep_idx=sep_idx, pad_idx=pad_idx)
    else:
        valid_len = valid_mask.sum(dim=1)
        sep_pos = torch.clamp(valid_len // 2, min=1, max=max(1, L - 2))
    return (positions - sep_pos.unsqueeze(1)).float() * valid_mask.float()


class FlexibleDisentangledMultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, max_len=512, dropout=0.0, use_c2p=True, use_p2c=True, use_sepdist=False):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_relative_positions = max_len
        self.use_c2p = use_c2p
        self.use_p2c = use_p2c
        self.use_sepdist = use_sepdist

        self.q_proj = nn.Linear(input_dim, embed_dim)
        self.k_proj = nn.Linear(input_dim, embed_dim)
        self.v_proj = nn.Linear(input_dim, embed_dim)

        self.rel_pos_emb = nn.Embedding(2 * max_len - 1, input_dim)
        self.pos_key_proj = nn.Linear(input_dim, embed_dim)
        self.pos_query_proj = nn.Linear(input_dim, embed_dim)

        if use_sepdist:
            self.sep_dist_emb = nn.Embedding(2 * max_len - 1, input_dim)
            self.sep_key_proj = nn.Linear(input_dim, embed_dim)

        self.o_proj = nn.Linear(embed_dim, input_dim)
        self.attn_drop = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.pos_key_proj.weight)
        nn.init.xavier_uniform_(self.pos_query_proj.weight)
        if self.use_sepdist:
            nn.init.xavier_uniform_(self.sep_key_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)

        self.q_proj.bias.data.zero_()
        self.k_proj.bias.data.zero_()
        self.v_proj.bias.data.zero_()
        self.pos_key_proj.bias.data.zero_()
        self.pos_query_proj.bias.data.zero_()
        if self.use_sepdist:
            self.sep_key_proj.bias.data.zero_()
        self.o_proj.bias.data.zero_()

    def _shape(self, x, batch_size, seq_length):
        return x.view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _get_rel_pos_embeddings(self, seq_length, device):
        rel_pos = build_relative_positions(seq_length, device=device)
        rel_pos = rel_pos.clamp(
            min=-(self.max_relative_positions - 1),
            max=(self.max_relative_positions - 1),
        )
        rel_pos = rel_pos + (self.max_relative_positions - 1)
        return self.rel_pos_emb(rel_pos)

    def _get_sep_distance_embeddings(self, sep_distances: torch.Tensor):
        max_offset = self.max_relative_positions - 1
        sep_ids = sep_distances.clamp(min=-max_offset, max=max_offset).long() + max_offset
        return self.sep_dist_emb(sep_ids)

    def forward(self, x, mask=None, sep_distances=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        if mask is not None:
            mask = expand_mask(mask)

        q = self._shape(self.q_proj(x), batch_size, seq_length)
        k = self._shape(self.k_proj(x), batch_size, seq_length)
        v = self._shape(self.v_proj(x), batch_size, seq_length)

        rel_embeddings = self._get_rel_pos_embeddings(seq_length, x.device)
        pos_k = self.pos_key_proj(rel_embeddings)
        pos_q = self.pos_query_proj(rel_embeddings)
        pos_k = pos_k.view(seq_length, seq_length, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
        pos_q = pos_q.view(seq_length, seq_length, self.num_heads, self.head_dim).permute(2, 0, 1, 3)

        terms = [torch.matmul(q, k.transpose(-2, -1))]
        if self.use_c2p:
            terms.append(torch.einsum("bhid,hijd->bhij", q, pos_k))
        if self.use_p2c:
            terms.append(torch.einsum("hijd,bhjd->bhij", pos_q, k))
        if self.use_sepdist and sep_distances is not None:
            sep_embeddings = self._get_sep_distance_embeddings(sep_distances)
            sep_k = self._shape(self.sep_key_proj(sep_embeddings), batch_size, seq_length)
            terms.append(torch.matmul(q, sep_k.transpose(-2, -1)))

        scale_factor = float(len(terms))
        attn_logits = sum(terms) / math.sqrt(self.head_dim * scale_factor)

        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)

        attention = F.softmax(attn_logits, dim=-1)
        attention = self.attn_drop(attention)
        values = torch.matmul(attention, v)
        values = values.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)
        if return_attention:
            return o, attention
        return o


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, dim_feedforward, num_heads, max_len=512, dropout=0.0, act_fn=nn.ReLU, use_c2p=True, use_p2c=True, use_sepdist=False):
        super().__init__()
        self.self_attn = FlexibleDisentangledMultiheadAttention(
            input_dim=input_dim,
            embed_dim=input_dim,
            num_heads=num_heads,
            max_len=max_len,
            dropout=dropout,
            use_c2p=use_c2p,
            use_p2c=use_p2c,
            use_sepdist=use_sepdist,
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

    def forward(self, x, mask=None, sep_distances=None):
        attn_out = self.self_attn(x, mask=mask, sep_distances=sep_distances)
        x = self.norm1(x + self.drop(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.drop(ff_out))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None, sep_distances=None):
        for layer in self.layers:
            x = layer(x, mask=mask, sep_distances=sep_distances)
        return x


class BertFlexibleAttentionTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes=3,
        pad_idx=0,
        sep_idx=None,
        model_dim=256,
        num_heads=8,
        num_layers=4,
        ff_mult=2,
        dropout=0.1,
        max_len=512,
        pooling="mean",
        use_absolute_pos=False,
        use_c2p=True,
        use_p2c=True,
        use_sepdist=False,
    ):
        super().__init__()
        assert pooling in ("mean", "cls")
        self.pad_idx = pad_idx
        self.sep_idx = sep_idx
        self.pooling = pooling
        self.use_absolute_pos = use_absolute_pos

        self.embedding = nn.Embedding(vocab_size, model_dim, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(d_model=model_dim, max_len=max_len)
        self.transformer = TransformerEncoder(
            num_layers=num_layers,
            input_dim=model_dim,
            dim_feedforward=ff_mult * model_dim,
            num_heads=num_heads,
            max_len=max_len,
            dropout=dropout,
            act_fn=nn.ReLU,
            use_c2p=use_c2p,
            use_p2c=use_p2c,
            use_sepdist=use_sepdist,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        mask = (x != self.pad_idx)
        sep_distances = build_sep_distances(x, pad_idx=self.pad_idx, sep_idx=self.sep_idx)
        x = self.embedding(x)
        if self.use_absolute_pos:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask, sep_distances=sep_distances)

        if self.pooling == "mean":
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
            x = x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            x = x[:, 0]

        x = self.dropout(x)
        return self.fc_out(x)


class BertSepDistanceTransformer(BertFlexibleAttentionTransformer):
    def __init__(self, **kwargs):
        super().__init__(use_c2p=True, use_p2c=True, use_sepdist=True, **kwargs)


class BertRelSimpleTransformer(BertFlexibleAttentionTransformer):
    def __init__(self, **kwargs):
        super().__init__(use_c2p=True, use_p2c=False, use_sepdist=False, **kwargs)
