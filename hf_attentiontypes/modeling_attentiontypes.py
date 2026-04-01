import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from .configuration_attentiontypes import AttentionTypesConfig


def expand_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(1).unsqueeze(2)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim != 4:
        raise ValueError(f"Expected 2D, 3D, or 4D mask; got shape {tuple(mask.shape)}")
    return mask


def build_relative_positions(seq_len: int, device: torch.device) -> torch.Tensor:
    position_ids = torch.arange(seq_len, device=device)
    return position_ids[None, :] - position_ids[:, None]


class DisentangledMultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, max_len=512, dropout=0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_relative_positions = max_len

        self.q_proj = nn.Linear(input_dim, embed_dim)
        self.k_proj = nn.Linear(input_dim, embed_dim)
        self.v_proj = nn.Linear(input_dim, embed_dim)
        self.rel_pos_emb = nn.Embedding(2 * max_len - 1, input_dim)
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

        self.q_proj.bias.data.zero_()
        self.k_proj.bias.data.zero_()
        self.v_proj.bias.data.zero_()
        self.pos_key_proj.bias.data.zero_()
        self.pos_query_proj.bias.data.zero_()
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

    def forward(self, x, mask=None):
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

        c2c = torch.matmul(q, k.transpose(-2, -1))
        c2p = torch.einsum("bhid,hijd->bhij", q, pos_k)
        p2c = torch.einsum("hijd,bhjd->bhij", pos_q, k)

        attn_logits = (c2c + c2p + p2c) / math.sqrt(self.head_dim * 3.0)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)

        attention = F.softmax(attn_logits, dim=-1)
        attention = self.attn_drop(attention)

        values = torch.matmul(attention, v)
        values = values.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, self.embed_dim)
        return self.o_proj(values)


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, dim_feedforward, num_heads, max_len=512, dropout=0.0):
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
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, input_dim),
        )

    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, mask=mask)
        x = self.norm1(x + self.drop(attn_out))
        ff_out = self.ff(x)
        return self.norm2(x + self.drop(ff_out))


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class AttentionTypesForSequenceClassification(PreTrainedModel):
    config_class = AttentionTypesConfig
    base_model_prefix = "attentiontypes"
    main_input_name = "input_ids"

    def __init__(self, config: AttentionTypesConfig):
        super().__init__(config)
        if config.pooling not in ("mean", "cls"):
            raise ValueError("pooling must be 'mean' or 'cls'")

        self.pad_idx = config.pad_token_id
        self.pooling = config.pooling
        self.use_absolute_pos = config.use_absolute_pos

        self.embedding = nn.Embedding(config.vocab_size, config.model_dim, padding_idx=self.pad_idx)
        self.positional_encoding = PositionalEncoding(
            d_model=config.model_dim,
            max_len=config.max_position_embeddings,
        )
        self.transformer = TransformerEncoder(
            num_layers=config.num_layers,
            input_dim=config.model_dim,
            dim_feedforward=config.ff_mult * config.model_dim,
            num_heads=config.num_heads,
            max_len=config.max_position_embeddings,
            dropout=config.dropout,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.out = nn.Linear(config.model_dim, config.num_labels)

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if input_ids is None:
            raise ValueError("input_ids is required")

        if attention_mask is None:
            attention_mask = (input_ids != self.pad_idx).long()

        x = self.embedding(input_ids)
        if self.use_absolute_pos:
            x = self.positional_encoding(x)

        x = self.transformer(x, mask=attention_mask)

        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            denom = mask.sum(dim=1).clamp(min=1.0)
            x = (x * mask).sum(dim=1) / denom
        else:
            x = x[:, 0, :]

        logits = self.out(self.dropout(x))

        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                labels = labels.to(logits.device).view_as(logits).float()
                loss = F.binary_cross_entropy_with_logits(logits, labels)
            else:
                loss = F.cross_entropy(logits.view(-1, self.config.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)
