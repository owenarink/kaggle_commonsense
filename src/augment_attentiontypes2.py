import torch


def _eligible_token_mask(x_ids: torch.Tensor, pad_id: int, protected_ids=()):
    mask = x_ids != int(pad_id)
    for pid in protected_ids:
        mask &= x_ids != int(pid)
    return mask


def token_dropout(x_ids: torch.Tensor, pad_id: int, unk_id: int, p: float, protected_ids=()):
    if p <= 0:
        return x_ids
    x = x_ids.clone()
    mask = _eligible_token_mask(x, pad_id, protected_ids)
    mask &= torch.rand_like(x.float()) < p
    x[mask] = int(unk_id)
    return x


def span_mask_to_unk(
    x_ids: torch.Tensor,
    pad_id: int,
    unk_id: int,
    p: float,
    span_len: int = 2,
    protected_ids=(),
):
    if p <= 0 or span_len <= 0:
        return x_ids

    x = x_ids.clone()
    B, K, L = x.shape
    eligible = _eligible_token_mask(x, pad_id, protected_ids)

    for b in range(B):
        for k in range(K):
            if torch.rand((), device=x.device).item() >= p:
                continue
            valid = torch.nonzero(eligible[b, k], as_tuple=False).flatten()
            if valid.numel() == 0:
                continue
            start_idx = valid[torch.randint(0, valid.numel(), (1,), device=x.device)].item()
            end_idx = min(start_idx + span_len, L)
            span_mask = eligible[b, k, start_idx:end_idx]
            x[b, k, start_idx:end_idx][span_mask] = int(unk_id)
    return x


def adjacent_token_swap(x_ids: torch.Tensor, pad_id: int, p: float, protected_ids=()):
    if p <= 0:
        return x_ids

    x = x_ids.clone()
    B, K, L = x.shape
    eligible = _eligible_token_mask(x, pad_id, protected_ids)

    for b in range(B):
        for k in range(K):
            pos = 0
            while pos < L - 1:
                if not (eligible[b, k, pos] and eligible[b, k, pos + 1]):
                    pos += 1
                    continue
                if torch.rand((), device=x.device).item() < p:
                    tmp = x[b, k, pos].item()
                    x[b, k, pos] = x[b, k, pos + 1]
                    x[b, k, pos + 1] = tmp
                    pos += 2
                else:
                    pos += 1
    return x


def augment_grouped_bbpe(
    x_ids: torch.Tensor,
    pad_id: int,
    unk_id: int,
    protected_ids=(),
    token_dropout_p: float = 0.08,
    swap_p: float = 0.03,
    span_mask_p: float = 0.12,
    span_len: int = 2,
):
    x = token_dropout(
        x_ids,
        pad_id=pad_id,
        unk_id=unk_id,
        p=token_dropout_p,
        protected_ids=protected_ids,
    )
    x = adjacent_token_swap(
        x,
        pad_id=pad_id,
        p=swap_p,
        protected_ids=protected_ids,
    )
    x = span_mask_to_unk(
        x,
        pad_id=pad_id,
        unk_id=unk_id,
        p=span_mask_p,
        span_len=span_len,
        protected_ids=protected_ids,
    )
    return x
