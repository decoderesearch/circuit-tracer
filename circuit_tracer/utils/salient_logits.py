import torch


@torch.no_grad()
def compute_salient_logits(
    logits: torch.Tensor,
    unembed_proj: torch.Tensor,
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pick the smallest logit set whose cumulative prob >= *desired_logit_prob*.

    Args:
        logits: ``(d_vocab,)`` vector (single position).
        unembed_proj: ``(d_model, d_vocab)`` unembedding matrix.
        max_n_logits: Hard cap *k*.
        desired_logit_prob: Cumulative probability threshold *p*.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            * logit_indices - ``(k,)`` vocabulary ids.
            * logit_probs   - ``(k,)`` softmax probabilities.
            * demeaned_vecs - ``(k, d_model)`` unembedding columns, demeaned.
    """

    probs = torch.softmax(logits, dim=-1)
    top_p, top_idx = torch.topk(probs, max_n_logits)
    cutoff = int(torch.searchsorted(torch.cumsum(top_p, 0), desired_logit_prob)) + 1
    top_p, top_idx = top_p[:cutoff], top_idx[:cutoff]

    # unembed_proj can be presented as (d_model, d_vocab) or its transpose (d_vocab, d_model).
    # We determine which axis corresponds to the vocabulary by matching against the logits length.

    if unembed_proj.shape[0] == logits.shape[0]:
        # Shape is (d_vocab, d_model) – first axis is vocabulary.
        cols = unembed_proj[top_idx]  # (k, d_model)
        demean = unembed_proj.mean(dim=0, keepdim=True)  # (1, d_model)
        demeaned_vecs = cols - demean  # (k, d_model)

    else:
        # Shape is (d_model, d_vocab) – second axis is vocabulary.
        cols = unembed_proj[:, top_idx]  # (d_model, k)
        demean = unembed_proj.mean(dim=-1, keepdim=True)  # (d_model, 1)
        demeaned_vecs = (cols - demean).T  # (k, d_model)

    return top_idx, top_p, demeaned_vecs
