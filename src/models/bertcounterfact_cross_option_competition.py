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

    false_mask = (positions < sep_pos_exp) & valid_mask
    option_mask = (positions > sep_pos_exp) & valid_mask

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
    Single-option counterfactual repair module.
    Input:
        x: [B, L, D]
        x_ids: [B, L]
    Output:
        fused_vec: [B, D]
        extras: dict
    """
    def __init__(self, model_dim, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim

        self.anomaly_proj = nn.Linear(model_dim, 1)

        self.support_q = nn.Linear(model_dim, model_dim)
        self.support_k = nn.Linear(model_dim, model_dim)

        self.conflict_q = nn.Linear(model_dim, model_dim)
        self.conflict_k = nn.Linear(model_dim, model_dim)

        self.repair_q = nn.Linear(model_dim, model_dim)
        self.repair_k = nn.Linear(model_dim, model_dim)

        self.fuse = nn.Sequential(
            nn.Linear(3 * model_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )

        self.norm = nn.LayerNorm(model_dim)

    def masked_softmax(self, logits, mask, dim=-1):
        logits = logits.masked_fill(~mask, -9e15)
        return F.softmax(logits, dim=dim)

    def forward(self, x, x_ids, pad_idx, sep_idx=None):
        B, L, D = x.shape
        false_mask, option_mask, valid_mask, sep_pos = build_segment_masks(
            x_ids=x_ids,
            pad_idx=pad_idx,
            sep_idx=sep_idx,
        )

        anomaly_logits = self.anomaly_proj(x).squeeze(-1)
        anomaly_logits = anomaly_logits.masked_fill(~false_mask, -9e15)
        anomaly_gate = F.softmax(anomaly_logits, dim=1)
        anomaly_gate = anomaly_gate * false_mask.float()
        anomaly_gate = anomaly_gate / anomaly_gate.sum(dim=1, keepdim=True).clamp(min=1e-8)

        q_sup = self.support_q(x)
        k_sup = self.support_k(x)

        q_con = self.conflict_q(x)
        k_con = self.conflict_k(x)

        q_rep = self.repair_q(x)
        k_rep = self.repair_k(x)

        support_scores = torch.matmul(q_sup, k_sup.transpose(-2, -1)) / math.sqrt(D)
        conflict_scores = torch.matmul(q_con, k_con.transpose(-2, -1)) / math.sqrt(D)
        repair_scores = torch.matmul(q_rep, k_rep.transpose(-2, -1)) / math.sqrt(D)

        pair_mask = false_mask.unsqueeze(2) & option_mask.unsqueeze(1)

        support_attn = self.masked_softmax(support_scores, pair_mask, dim=-1)
        conflict_signed = torch.tanh(conflict_scores)
        repair_logits = repair_scores + conflict_signed
        repair_attn = self.masked_softmax(repair_logits, pair_mask, dim=-1)

        repair_token_vec = torch.matmul(repair_attn, x)
        support_token_vec = torch.matmul(support_attn, x)

        anomaly_false_vec = torch.sum(anomaly_gate.unsqueeze(-1) * x, dim=1)
        weighted_repair_vec = torch.sum(anomaly_gate.unsqueeze(-1) * repair_token_vec, dim=1)
        weighted_support_vec = torch.sum(anomaly_gate.unsqueeze(-1) * support_token_vec, dim=1)

        fused = torch.cat([anomaly_false_vec, weighted_repair_vec, weighted_support_vec], dim=-1)
        fused = self.fuse(fused)
        fused = self.norm(fused)

        # token-wise repair strength per false token
        token_repair_strength = (repair_attn * pair_mask.float()).sum(dim=-1) * false_mask.float()

        extras = {
            "false_mask": false_mask,
            "option_mask": option_mask,
            "anomaly_gate": anomaly_gate,
            "support_attn": support_attn,
            "repair_attn": repair_attn,
            "conflict_signed": conflict_signed,
            "sep_pos": sep_pos,
            "token_repair_strength": token_repair_strength,  # [B, L]
        }
        return fused, extras


class CrossOptionCompetition(nn.Module):
    """
    Competition across K options for each question.

    Input:
        pooled_vec: [B, K, D]
        repair_vec: [B, K, D]
        anomaly_gate: [B, K, L]
        token_repair_strength: [B, K, L]
        false_mask: [B, K, L]

    Output:
        competed_vec: [B, K, D]
        competition_scores: [B, K]
        extras: dict
    """
    def __init__(self, model_dim, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim

        self.option_proj = nn.Linear(2 * model_dim, model_dim)
        self.comp_vector = nn.Linear(model_dim, 1)
        self.expl_score = nn.Sequential(
            nn.Linear(4 * model_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )

        self.token_comp_proj = nn.Linear(2 * model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, pooled_vec, repair_vec, anomaly_gate, token_repair_strength, false_mask):
        B, K, D = pooled_vec.shape
        L = anomaly_gate.size(-1)

        option_repr = torch.cat([pooled_vec, repair_vec], dim=-1)   # [B, K, 2D]
        option_repr = self.option_proj(option_repr)                 # [B, K, D]

        # One corruption should be shared across the three options.
        shared_false_mask = false_mask[:, 0, :]
        shared_anomaly = anomaly_gate.mean(dim=1)
        shared_anomaly = shared_anomaly * shared_false_mask.float()
        shared_anomaly = shared_anomaly / shared_anomaly.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        shared_question = pooled_vec.mean(dim=1, keepdim=True).expand(B, K, D)

        explanation_features = torch.cat(
            [
                shared_question,
                repair_vec,
                torch.abs(shared_question - repair_vec),
                shared_question * repair_vec,
            ],
            dim=-1,
        )
        explanation_scores = self.expl_score(explanation_features).squeeze(-1)      # [B, K]

        # Global option competition score
        raw_option_scores = self.comp_vector(torch.tanh(option_repr)).squeeze(-1)   # [B, K]

        # Token-level repair vote:
        # anomaly-weighted repair score per token, per option, using the shared corruption signal
        token_vote = shared_anomaly.unsqueeze(1) * token_repair_strength * false_mask.float()  # [B, K, L]

        # Compete across options for each false token
        option_token_comp = F.softmax(token_vote, dim=1)                             # [B, K, L]

        # Aggregate token competition into option score
        token_comp_score = option_token_comp.sum(dim=-1)                             # [B, K]

        # Combine global and token competition
        competition_scores = raw_option_scores + token_comp_score + explanation_scores  # [B, K]

        # Rival-average representation
        mean_all = option_repr.mean(dim=1, keepdim=True)                             # [B, 1, D]
        rival_mean = (K * mean_all - option_repr) / max(K - 1, 1)                   # [B, K, D]

        # Difference-from-rivals feature
        diff_repr = option_repr - rival_mean                                         # [B, K, D]

        # Token-conditioned competition summary
        token_context = torch.cat([option_repr, diff_repr], dim=-1)                  # [B, K, 2D]
        token_context = self.token_comp_proj(token_context)                          # [B, K, D]

        # Weight each option representation by its competition strength
        comp_weight = F.softmax(competition_scores, dim=1).unsqueeze(-1)             # [B, K, 1]
        competed_vec = option_repr + diff_repr + token_context * comp_weight
        competed_vec = self.norm(self.dropout(competed_vec))

        extras = {
            "raw_option_scores": raw_option_scores,
            "shared_anomaly": shared_anomaly,
            "explanation_scores": explanation_scores,
            "token_vote": token_vote,
            "option_token_comp": option_token_comp,
            "token_comp_score": token_comp_score,
            "competition_scores": competition_scores,
            "rival_mean": rival_mean,
            "diff_repr": diff_repr,
        }
        return competed_vec, competition_scores, extras


class BertCounterFactCrossOpitionCompetitionTransformer(nn.Module):
    """
    DeBERTa-style encoder + Counterfactual Repair Attention + Cross-Option Competition.

    Input:
        x_ids: [B, K, L]  (usually K=3)

    Output:
        logits: [B, K]

    This model is designed for grouped multiple-choice input so that
    the options compete inside the model instead of only in the loss.
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

        self.cross_option = CrossOptionCompetition(
            model_dim=model_dim,
            dropout=dropout,
        )

        self.final_proj = nn.Sequential(
            nn.Linear(3 * model_dim, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(model_dim, 1)

    def _pool_encoder(self, x, pad_mask):
        if self.pooling == "mean":
            return masked_mean(x, pad_mask, dim=1)
        return x[:, 0, :]

    def forward(self, x_ids, return_extras=False):
        """
        x_ids: [B, K, L]
        returns logits: [B, K]
        """
        if x_ids.ndim != 3:
            raise ValueError("BertCounterFactCrossOpitionCompetitionTransformer expects x_ids with shape [B, K, L].")

        B, K, L = x_ids.shape
        x_flat = x_ids.view(B * K, L)

        pad_mask = (x_flat != self.pad_idx)

        x = self.embedding(x_flat)  # [B*K, L, D]

        if self.use_absolute_pos:
            x = self.positional_encoding(x)

        x = self.transformer(x, mask=pad_mask)

        pooled_vec = self._pool_encoder(x, pad_mask)  # [B*K, D]
        repair_vec, cf_extras = self.counterfact(
            x=x,
            x_ids=x_flat,
            pad_idx=self.pad_idx,
            sep_idx=self.sep_idx,
        )

        pooled_vec = pooled_vec.view(B, K, -1)
        repair_vec = repair_vec.view(B, K, -1)

        anomaly_gate = cf_extras["anomaly_gate"].view(B, K, L)
        false_mask = cf_extras["false_mask"].view(B, K, L)
        token_repair_strength = cf_extras["token_repair_strength"].view(B, K, L)

        competed_vec, competition_scores, comp_extras = self.cross_option(
            pooled_vec=pooled_vec,
            repair_vec=repair_vec,
            anomaly_gate=anomaly_gate,
            token_repair_strength=token_repair_strength,
            false_mask=false_mask,
        )

        combined_vec = torch.cat([pooled_vec, repair_vec, competed_vec], dim=-1)  # [B, K, 3D]
        combined_vec = self.final_proj(combined_vec)
        combined_vec = self.final_norm(combined_vec)
        combined_vec = self.dropout(combined_vec)

        logits = self.out(combined_vec).squeeze(-1) + competition_scores  # [B, K]

        if return_extras:
            extras = {
                "counterfact": {
                    "anomaly_gate": anomaly_gate,
                    "false_mask": false_mask,
                    "token_repair_strength": token_repair_strength,
                },
                "competition": comp_extras,
            }
            return logits, extras

        return logits

    @torch.no_grad()
    def get_attention_maps(self, x_ids):
        if x_ids.ndim != 3:
            raise ValueError("get_attention_maps expects x_ids with shape [B, K, L].")

        B, K, L = x_ids.shape
        x_flat = x_ids.view(B * K, L)
        pad_mask = (x_flat != self.pad_idx)
        x = self.embedding(x_flat)

        if self.use_absolute_pos:
            x = self.positional_encoding(x)

        maps = self.transformer.get_attention_maps(x, mask=pad_mask)
        return maps

    @torch.no_grad()
    def get_counterfactual_maps(self, x_ids):
        if x_ids.ndim != 3:
            raise ValueError("get_counterfactual_maps expects x_ids with shape [B, K, L].")

        B, K, L = x_ids.shape
        x_flat = x_ids.view(B * K, L)
        pad_mask = (x_flat != self.pad_idx)

        x = self.embedding(x_flat)
        if self.use_absolute_pos:
            x = self.positional_encoding(x)

        x = self.transformer(x, mask=pad_mask)
        _, cf_extras = self.counterfact(
            x=x,
            x_ids=x_flat,
            pad_idx=self.pad_idx,
            sep_idx=self.sep_idx,
        )

        out = {}
        for key, value in cf_extras.items():
            if torch.is_tensor(value) and value.size(0) == B * K:
                out[key] = value.view(B, K, *value.shape[1:])
            else:
                out[key] = value
        return out


if __name__ == "__main__":
    B, K, L = 2, 3, 8
    vocab_size = 100
    pad_idx = 0
    sep_idx = 2

    x = torch.randint(3, vocab_size, (B, K, L))
    x[:, :, -2:] = pad_idx
    x[:, :, 3] = sep_idx

    clf = BertCounterFactCrossOpitionCompetitionTransformer(
        vocab_size=vocab_size,
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
    print("BertCounterFactCrossOpitionCompetitionTransformer logits:", y.shape)  # [B, K]
    print("anomaly_gate:", extras["counterfact"]["anomaly_gate"].shape)          # [B, K, L]
    print("token_vote:", extras["competition"]["token_vote"].shape)              # [B, K, L]
