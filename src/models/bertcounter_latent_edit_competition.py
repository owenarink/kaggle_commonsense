import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def expand_mask(mask: torch.Tensor) -> torch.Tensor:
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
    position_ids = torch.arange(seq_len, device=device)
    rel_pos = position_ids[None, :] - position_ids[:, None]
    return rel_pos


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int):
    m = mask.unsqueeze(-1).float()
    denom = m.sum(dim=dim).clamp(min=1.0)
    return (x * m).sum(dim=dim) / denom


def first_sep_index(x_ids: torch.Tensor, sep_idx: int, pad_idx: int) -> torch.Tensor:
    B, L = x_ids.shape
    sep_mask = (x_ids == sep_idx)
    has_sep = sep_mask.any(dim=1)

    idxs = torch.argmax(sep_mask.int(), dim=1)

    valid_len = (x_ids != pad_idx).sum(dim=1)
    fallback = torch.clamp(valid_len // 2, min=1, max=max(1, L - 2))

    idxs = torch.where(has_sep, idxs, fallback)
    return idxs


def build_segment_masks(x_ids: torch.Tensor, pad_idx: int, sep_idx: int = None):
    B, L = x_ids.shape
    valid_mask = (x_ids != pad_idx)
    positions = torch.arange(L, device=x_ids.device).unsqueeze(0).expand(B, L)

    if sep_idx is not None:
        sep_pos = first_sep_index(x_ids, sep_idx=sep_idx, pad_idx=pad_idx)
    else:
        valid_len = valid_mask.sum(dim=1)
        sep_pos = torch.clamp(valid_len // 2, min=1, max=max(1, L - 2))

    sep_pos_exp = sep_pos.unsqueeze(1)
    valid_len = valid_mask.sum(dim=1, keepdim=True)
    eos_pos_exp = torch.clamp(valid_len - 1, min=0)

    # The task is about repairing content in the false sentence, not special tokens.
    false_mask = (positions > 0) & (positions < sep_pos_exp) & valid_mask
    option_mask = (positions > sep_pos_exp) & (positions < eos_pos_exp) & valid_mask

    return false_mask, option_mask, valid_mask, sep_pos


class DisentangledMultiheadAttention(nn.Module):
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
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

    @torch.no_grad()
    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for layer in self.layers:
            attn_out, attn_map = layer.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = layer.norm1(x + layer.drop(attn_out))
            ff_out = layer.ff(x)
            x = layer.norm2(x + layer.drop(ff_out))
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


class LatentEditCompetition(nn.Module):
    """
    Latent edit competition over grouped options.

    Input:
        x:     [B, K, L, D]
        x_ids: [B, K, L]

    Output:
        edit_vec: [B, K, D]
        option_scores: [B, K]
        extras: dict
    """ 

    def __init__(self, model_dim, dropout=0.1, edit_minimality_weight=0.10):
            super().__init__()
            self.model_dim = model_dim
            self.edit_minimality_weight = edit_minimality_weight

            # Localize the likely corruption on the false side.
            self.anomaly_proj = nn.Linear(model_dim, 1)

            # Build an explanation-focused option summary conditioned on the anomaly.
            self.option_attn_q = nn.Linear(model_dim, model_dim)
            self.option_attn_k = nn.Linear(model_dim, model_dim)

            # One shared latent correction vector for the false sentence.
            self.delta_proj = nn.Linear(model_dim, model_dim)

            # Compare repaired false-side content to an explanation summary.
            self.edit_q = nn.Linear(model_dim, model_dim)
            self.edit_k = nn.Linear(model_dim, model_dim)

            # Score whether an option explains why the false-side content is wrong.
            self.explanation_score = nn.Sequential(
                nn.Linear(4 * model_dim, model_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(model_dim, 1),
            )

            # Explanation-first fusion for the final classifier state.
            self.fuse = nn.Sequential(
                nn.Linear(3 * model_dim, model_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(model_dim, model_dim),
            )
            self.norm = nn.LayerNorm(model_dim)

    def _masked_softmax(self, logits, mask, dim=-1):
        logits = logits.masked_fill(~mask, -9e15)
        return F.softmax(logits, dim=dim)

    def forward(self, x, x_ids, pad_idx, sep_idx=None):
        B, K, L, D = x.shape

        # flatten to reuse segment builder
        x_flat = x.view(B * K, L, D)
        x_ids_flat = x_ids.view(B * K, L)

        false_mask, option_mask, _, sep_pos = build_segment_masks(
            x_ids=x_ids_flat,
            pad_idx=pad_idx,
            sep_idx=sep_idx,
        )

        false_mask = false_mask.view(B, K, L)
        option_mask = option_mask.view(B, K, L)
        shared_false_mask = false_mask[:, 0, :]

        # The corruption is a property of the false sentence, not of any one option.
        shared_false_tokens = x.mean(dim=1)                                           # [B, L, D]

        # Infer one shared anomaly distribution over the false sentence.
        anomaly_logits = self.anomaly_proj(shared_false_tokens).squeeze(-1)            # [B, L]
        anomaly_logits = anomaly_logits.masked_fill(~shared_false_mask, -9e15)

        anomaly_gate = F.softmax(anomaly_logits, dim=-1)                               # [B, L]
        anomaly_gate = anomaly_gate * shared_false_mask.float()
        anomaly_gate = anomaly_gate / anomaly_gate.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        shared_false_summary = torch.sum(
            anomaly_gate.unsqueeze(-1) * shared_false_tokens, dim=1
        )                                                                               # [B, D]
        shared_false_context = masked_mean(shared_false_tokens, shared_false_mask, dim=1)  # [B, D]

        false_summary = shared_false_summary.unsqueeze(1).expand(B, K, D)              # [B, K, D]
        false_context = shared_false_context.unsqueeze(1).expand(B, K, D)              # [B, K, D]
        option_summary = masked_mean(x, option_mask, dim=2)                             # [B, K, D]

        option_query = self.option_attn_q(false_summary).unsqueeze(2)                   # [B, K, 1, D]
        option_keys = self.option_attn_k(x)                                              # [B, K, L, D]
        option_attn_logits = (option_query * option_keys).sum(dim=-1) / math.sqrt(D)    # [B, K, L]
        option_attn = self._masked_softmax(option_attn_logits, option_mask, dim=-1)
        explanation_summary = torch.sum(option_attn.unsqueeze(-1) * x, dim=2)            # [B, K, D]

        # Keep one shared latent repair as a weak auxiliary signal only.
        shared_delta = torch.tanh(self.delta_proj(shared_false_summary))                 # [B, D]
        delta = shared_delta.unsqueeze(1).expand(B, K, D)                               # [B, K, D]
        edited_false_summary = false_summary + delta                                    # [B, K, D]
        shared_minimality_penalty = -self.edit_minimality_weight * (shared_delta ** 2).sum(dim=-1)
        minimality_penalty = shared_minimality_penalty.unsqueeze(1).expand(B, K)        # [B, K]

        false_q = self.edit_q(false_summary)
        edited_q = self.edit_q(edited_false_summary)
        explanation_k = self.edit_k(explanation_summary)

        base_alignment = (false_q * explanation_k).sum(dim=-1) / math.sqrt(D)          # [B, K]
        repaired_alignment = (edited_q * explanation_k).sum(dim=-1) / math.sqrt(D)     # [B, K]
        repair_gain = repaired_alignment - base_alignment

        explanation_features = torch.cat(
            [
                false_context,
                explanation_summary,
                torch.abs(false_context - explanation_summary),
                false_context * explanation_summary,
            ],
            dim=-1,
        )
        explanation_gain = self.explanation_score(explanation_features).squeeze(-1)      # [B, K]

        # The labels are explanation-like, so explanation evidence should dominate.
        raw_scores = explanation_gain + 0.15 * repair_gain + 0.05 * minimality_penalty
        option_scores = raw_scores - raw_scores.mean(dim=1, keepdim=True)
        option_competition = F.softmax(option_scores, dim=1)                            # [B, K]

        competition_summary = option_competition.unsqueeze(-1) * explanation_summary

        explanation_interaction = false_summary * explanation_summary
        edit_vec = torch.cat([false_summary, explanation_summary, explanation_interaction], dim=-1)
        edit_vec = self.fuse(edit_vec)
        edit_vec = self.norm(edit_vec)

        extras = {
            "false_mask": false_mask,
            "option_mask": option_mask,
            "anomaly_gate": anomaly_gate,
            "shared_false_mask": shared_false_mask,
            "shared_delta": shared_delta,
            "delta": delta,
            "shared_false_summary": shared_false_summary,
            "false_summary": false_summary,
            "false_context": false_context,
            "edited_false_summary": edited_false_summary,
            "option_summary": option_summary,
            "option_attn": option_attn,
            "explanation_summary": explanation_summary,
            "minimality_penalty": minimality_penalty,
            "option_competition": option_competition,
            "base_alignment": base_alignment,
            "repaired_alignment": repaired_alignment,
            "repair_gain": repair_gain,
            "explanation_gain": explanation_gain,
            "sep_pos": sep_pos.view(B, K),
        }
        return edit_vec, option_scores, extras


class BertCounterFactLatentEditCompetitionTransformer(nn.Module):
    """
    DeBERTa-style encoder + latent edit competition.

    Input:
        x_ids: [B, K, L]   (usually K=3)

    Output:
        logits: [B, K]
    """
    def __init__(
        self,
        vocab_size,
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
        edit_minimality_weight=0.10,
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

        self.latent_edit = LatentEditCompetition(
            model_dim=model_dim,
            dropout=dropout,
            edit_minimality_weight=edit_minimality_weight, 
        )

        self.final_proj = nn.Sequential(
            nn.Linear(4 * model_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(model_dim, 1)

    def _pool_encoder(self, x, pad_mask):
        if self.pooling == "mean":
            return masked_mean(x, pad_mask, dim=2)
        return x[:, :, 0, :]

    def forward(self, x_ids, return_extras=False):
        """
        x_ids: [B, K, L]
        returns logits: [B, K]
        """
        if x_ids.ndim != 3:
            raise ValueError("BertCounterFactLatentEditCompetitionTransformer expects x_ids with shape [B, K, L].")

        B, K, L = x_ids.shape
        x_flat = x_ids.view(B * K, L)
        pad_mask_flat = (x_flat != self.pad_idx)

        x = self.embedding(x_flat)  # [B*K, L, D]

        if self.use_absolute_pos:
            x = self.positional_encoding(x)

        x = self.transformer(x, mask=pad_mask_flat)
        x = x.view(B, K, L, -1)

        pad_mask = (x_ids != self.pad_idx)                                          # [B, K, L]
        pooled_vec = self._pool_encoder(x, pad_mask)                                # [B, K, D]

        edit_vec, option_scores, extras = self.latent_edit(
            x=x,
            x_ids=x_ids,
            pad_idx=self.pad_idx,
            sep_idx=self.sep_idx,
        )

        competition_feature = option_scores.unsqueeze(-1) * edit_vec
        combined_vec = torch.cat(
            [pooled_vec, edit_vec, pooled_vec - edit_vec, competition_feature],
            dim=-1,
        )  # [B, K, 4D]
        combined_vec = self.final_proj(combined_vec)
        combined_vec = self.final_norm(combined_vec)
        combined_vec = self.dropout(combined_vec)

        logits = self.out(combined_vec).squeeze(-1)   # [B, K]

        if return_extras:
            return logits, extras
        return logits

    @torch.no_grad()
    def get_attention_maps(self, x_ids):
        if x_ids.ndim != 3:
            raise ValueError("get_attention_maps expects x_ids with shape [B, K, L].")

        B, K, L = x_ids.shape
        x_flat = x_ids.view(B * K, L)
        pad_mask_flat = (x_flat != self.pad_idx)

        x = self.embedding(x_flat)
        if self.use_absolute_pos:
            x = self.positional_encoding(x)

        return self.transformer.get_attention_maps(x, mask=pad_mask_flat)

    @torch.no_grad()
    def get_latent_edit_maps(self, x_ids):
        if x_ids.ndim != 3:
            raise ValueError("get_latent_edit_maps expects x_ids with shape [B, K, L].")

        B, K, L = x_ids.shape
        x_flat = x_ids.view(B * K, L)
        pad_mask_flat = (x_flat != self.pad_idx)

        x = self.embedding(x_flat)
        if self.use_absolute_pos:
            x = self.positional_encoding(x)

        x = self.transformer(x, mask=pad_mask_flat)
        x = x.view(B, K, L, -1)

        _, _, extras = self.latent_edit(
            x=x,
            x_ids=x_ids,
            pad_idx=self.pad_idx,
            sep_idx=self.sep_idx,
        )
        return extras


if __name__ == "__main__":
    B, K, L = 2, 3, 8
    vocab_size = 100
    pad_idx = 0
    sep_idx = 2

    x = torch.randint(3, vocab_size, (B, K, L))
    x[:, :, -2:] = pad_idx
    x[:, :, 3] = sep_idx

    clf = BertCounterFactLatentEditCompetitionTransformer(
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        sep_idx=sep_idx,
        model_dim=64,
        num_heads=4,
        num_layers=2,
        max_len=128,
        pooling="mean",
        use_absolute_pos=False,
        edit_minimality_weight=0.05,
    )

    y, extras = clf(x, return_extras=True)
    print("BertCounterFactLatentEditCompetitionTransformer logits:", y.shape)  # [B, K]
    print("anomaly_gate:", extras["anomaly_gate"].shape)                       # [B, K, L]
    print("delta:", extras["delta"].shape)                                     # [B, K, D]
    print("option_competition:", extras["option_competition"].shape)           # [B, K]
