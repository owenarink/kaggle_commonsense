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
    if mask.ndim == 2:
        mask = mask.unsqueeze(1).unsqueeze(2)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
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


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int):
    """
    x:    [..., L, D]
    mask: [..., L] bool
    """
    m = mask.unsqueeze(-1).float()
    denom = m.sum(dim=dim).clamp(min=1.0)
    return (x * m).sum(dim=dim) / denom


def first_sep_index(x_ids: torch.Tensor, sep_idx: int, pad_idx: int) -> torch.Tensor:
    """
    Find first occurrence of sep_idx in each row.
    If none exists, fall back to roughly half of the non-pad length.
    Returns shape [B] with integer split positions.
    """
    B, L = x_ids.shape
    device = x_ids.device

    sep_mask = (x_ids == sep_idx)
    has_sep = sep_mask.any(dim=1)

    # default = first sep
    idxs = torch.argmax(sep_mask.int(), dim=1)

    # fallback = half of valid length
    valid_len = (x_ids != pad_idx).sum(dim=1)
    fallback = torch.clamp(valid_len // 2, min=1, max=max(1, L - 2))

    idxs = torch.where(has_sep, idxs, fallback)
    return idxs


def build_segment_masks(x_ids: torch.Tensor, pad_idx: int, sep_idx: int = None):
    """
    Builds:
      false_mask: tokens before first SEP
      option_mask: tokens after first SEP and before PAD
      valid_mask: all non-pad tokens

    If sep_idx is None, uses a rough half-split over valid tokens.
    """
    B, L = x_ids.shape
    device = x_ids.device
    valid_mask = (x_ids != pad_idx)

    positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

    if sep_idx is not None:
        sep_pos = first_sep_index(x_ids, sep_idx=sep_idx, pad_idx=pad_idx)
    else:
        valid_len = valid_mask.sum(dim=1)
        sep_pos = torch.clamp(valid_len // 2, min=1, max=max(1, L - 2))

    sep_pos_exp = sep_pos.unsqueeze(1)

    false_mask = (positions < sep_pos_exp) & valid_mask
    option_mask = (positions > sep_pos_exp) & valid_mask

    return false_mask, option_mask, valid_mask, sep_pos


class DisentangledMultiheadAttention(nn.Module):
    """
    Simplified DeBERTa-style disentangled self-attention.

    Attention score = content->content + content->position + position->content
    """
    def __init__(self, input_dim, embed_dim, num_heads, max_len=512, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_len = max_len
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

        self.q_proj.bias.data.fill_(0)
        self.k_proj.bias.data.fill_(0)
        self.v_proj.bias.data.fill_(0)
        self.pos_key_proj.bias.data.fill_(0)
        self.pos_query_proj.bias.data.fill_(0)
        self.o_proj.bias.data.fill_(0)

    def _shape(self, x, batch_size, seq_length):
        return x.view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _get_rel_pos_embeddings(self, seq_length, device):
        rel_pos = build_relative_positions(seq_length, device=device)
        rel_pos = rel_pos.clamp(
            min=-(self.max_relative_positions - 1),
            max=(self.max_relative_positions - 1),
        )
        rel_pos = rel_pos + (self.max_relative_positions - 1)
        rel_embeddings = self.rel_pos_emb(rel_pos)
        return rel_embeddings

    def forward(self, x, mask=None, return_attention=False):
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

        scale_factor = 3.0
        attn_logits = (c2c + c2p + p2c) / math.sqrt(self.head_dim * scale_factor)

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


class CounterfactualRepairAttention(nn.Module):
    """
    Task-specific layer for inputs of the form:

        FalseSent <sep> Option

    Steps:
    1. Find false-sentence tokens and option tokens
    2. Predict anomaly gate over false-sentence tokens
    3. Compute support / conflict / repair attention from false -> option
    4. Build a repair summary vector

    Output:
      fused_vec: [B, D]
      extras: dict of interpretable tensors
    """
    def __init__(self, model_dim, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim

        # anomaly / brokenness score on false-sentence tokens
        self.anomaly_proj = nn.Linear(model_dim, 1)

        # support projections
        self.support_q = nn.Linear(model_dim, model_dim)
        self.support_k = nn.Linear(model_dim, model_dim)

        # conflict projections
        self.conflict_q = nn.Linear(model_dim, model_dim)
        self.conflict_k = nn.Linear(model_dim, model_dim)

        # repair projections
        self.repair_q = nn.Linear(model_dim, model_dim)
        self.repair_k = nn.Linear(model_dim, model_dim)

        # fuse token-level repair evidence back into a vector
        self.fuse = nn.Sequential(
            nn.Linear(3 * model_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )

        self.norm = nn.LayerNorm(model_dim)
        self.drop = nn.Dropout(dropout)

    def masked_softmax(self, logits, mask, dim=-1):
        logits = logits.masked_fill(~mask, -9e15)
        return F.softmax(logits, dim=dim)

    def forward(self, x, x_ids, pad_idx, sep_idx=None):
        """
        x:     [B, L, D]
        x_ids: [B, L]
        """
        B, L, D = x.shape
        false_mask, option_mask, valid_mask, sep_pos = build_segment_masks(
            x_ids=x_ids,
            pad_idx=pad_idx,
            sep_idx=sep_idx,
        )

        # [B, L]
        anomaly_logits = self.anomaly_proj(x).squeeze(-1)

        # only meaningful on false-sentence tokens
        anomaly_logits = anomaly_logits.masked_fill(~false_mask, -9e15)
        anomaly_gate = F.softmax(anomaly_logits, dim=1)  # [B, L]
        anomaly_gate = anomaly_gate * false_mask.float()
        anomaly_gate = anomaly_gate / anomaly_gate.sum(dim=1, keepdim=True).clamp(min=1e-8)

        # projected reps
        q_sup = self.support_q(x)
        k_sup = self.support_k(x)

        q_con = self.conflict_q(x)
        k_con = self.conflict_k(x)

        q_rep = self.repair_q(x)
        k_rep = self.repair_k(x)

        # false -> option pairwise scores
        # [B, L, L]
        support_scores = torch.matmul(q_sup, k_sup.transpose(-2, -1)) / math.sqrt(D)
        conflict_scores = torch.matmul(q_con, k_con.transpose(-2, -1)) / math.sqrt(D)
        repair_scores = torch.matmul(q_rep, k_rep.transpose(-2, -1)) / math.sqrt(D)

        pair_mask = false_mask.unsqueeze(2) & option_mask.unsqueeze(1)  # [B, L, L]

        # Support: normal positive attention
        support_attn = self.masked_softmax(support_scores, pair_mask, dim=-1)

        # Conflict: signed incompatibility
        # Use tanh so high positive/negative values can matter
        conflict_signed = torch.tanh(conflict_scores)

        # Repair: anomaly-gated false-token query attending into option
        repair_logits = repair_scores + conflict_signed
        repair_attn = self.masked_softmax(repair_logits, pair_mask, dim=-1)

        # summarize option content each false token uses as repair evidence
        repair_token_vec = torch.matmul(repair_attn, x)  # [B, L, D]
        support_token_vec = torch.matmul(support_attn, x)  # [B, L, D]

        # anomaly-weighted false sentence summary
        anomaly_false_vec = torch.sum(anomaly_gate.unsqueeze(-1) * x, dim=1)  # [B, D]

        # anomaly-weighted repair/support summaries
        weighted_repair_vec = torch.sum(anomaly_gate.unsqueeze(-1) * repair_token_vec, dim=1)  # [B, D]
        weighted_support_vec = torch.sum(anomaly_gate.unsqueeze(-1) * support_token_vec, dim=1)  # [B, D]

        fused = torch.cat([anomaly_false_vec, weighted_repair_vec, weighted_support_vec], dim=-1)
        fused = self.fuse(fused)
        fused = self.norm(fused)

        extras = {
            "false_mask": false_mask,
            "option_mask": option_mask,
            "anomaly_gate": anomaly_gate,
            "support_attn": support_attn,
            "repair_attn": repair_attn,
            "conflict_signed": conflict_signed,
            "sep_pos": sep_pos,
        }
        return fused, extras


class BertCounterFactTransformer(nn.Module):
    """
    DeBERTa-style encoder + Counterfactual Repair Attention.

    Intended for pair inputs shaped like:
        FalseSent <sep> Option

    For your training pipeline:
    - if num_classes=1, this acts as a scorer per option
    - if num_classes=3, this acts as a direct classifier

    The final representation combines:
    - base pooled sequence vector
    - counterfactual repair vector
    - anomaly-weighted false-sentence reasoning
    """
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
        pooling="mean",   # "mean" or "cls"
        use_absolute_pos=False,
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
        )

        self.counterfact = CounterfactualRepairAttention(
            model_dim=model_dim,
            dropout=dropout,
        )

        # combine pooled encoder vector + repair vector
        self.final_proj = nn.Sequential(
            nn.Linear(2 * model_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )

        self.final_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(model_dim, num_classes)

    def _pool_encoder(self, x, pad_mask):
        if self.pooling == "mean":
            return masked_mean(x, pad_mask, dim=1)
        return x[:, 0, :]

    def forward(self, x_ids, return_extras=False):
        pad_mask = (x_ids != self.pad_idx)  # [B, L]

        x = self.embedding(x_ids)  # [B, L, D]

        if self.use_absolute_pos:
            x = self.positional_encoding(x)

        x = self.transformer(x, mask=pad_mask)

        pooled_vec = self._pool_encoder(x, pad_mask)  # [B, D]
        repair_vec, extras = self.counterfact(
            x=x,
            x_ids=x_ids,
            pad_idx=self.pad_idx,
            sep_idx=self.sep_idx,
        )

        final_vec = torch.cat([pooled_vec, repair_vec], dim=-1)
        final_vec = self.final_proj(final_vec)
        final_vec = self.final_norm(final_vec)
        final_vec = self.dropout(final_vec)

        logits = self.out(final_vec)

        if return_extras:
            return logits, extras
        return logits

    @torch.no_grad()
    def get_attention_maps(self, x_ids):
        pad_mask = (x_ids != self.pad_idx).to(x_ids.device)
        x = self.embedding(x_ids)
        if self.use_absolute_pos:
            x = self.positional_encoding(x)
        return self.transformer.get_attention_maps(x, mask=pad_mask)

    @torch.no_grad()
    def get_counterfactual_maps(self, x_ids):
        pad_mask = (x_ids != self.pad_idx)
        x = self.embedding(x_ids)
        if self.use_absolute_pos:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=pad_mask)
        _, extras = self.counterfact(
            x=x,
            x_ids=x_ids,
            pad_idx=self.pad_idx,
            sep_idx=self.sep_idx,
        )
        return extras


if __name__ == "__main__":
    B, L = 2, 8
    vocab_size = 100
    pad_idx = 0
    sep_idx = 2

    x = torch.randint(3, vocab_size, (B, L))
    x[:, -2:] = pad_idx
    x[:, 3] = sep_idx

    clf = BertCounterfactTransformer(
        vocab_size=vocab_size,
        num_classes=1,
        pad_idx=pad_idx,
        sep_idx=sep_idx,
        model_dim=64,
        num_heads=4,
        num_layers=2,
        max_len=128,
        pooling="mean",
        use_absolute_pos=False,
    )

    y, extras = clf(x, return_extras=True)
    print("BertCounterfactTransformer logits:", y.shape)   # [B, 1]
    print("anomaly_gate:", extras["anomaly_gate"].shape)   # [B, L]
    print("repair_attn:", extras["repair_attn"].shape)     # [B, L, L]
