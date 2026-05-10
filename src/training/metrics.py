from __future__ import annotations

from typing import List

import torch


def token_accuracy(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> float:
    """
    Token-level accuracy ignoring PAD tokens.
    logits: [B, T, V]
    targets: [B, T]
    """
    preds = logits.argmax(dim=-1)  # [B, T]
    mask = targets.ne(pad_id)      # True where not PAD

    correct = (preds.eq(targets) & mask).sum().item()
    total = mask.sum().item()

    return float(correct) / float(total) if total > 0 else 0.0


def exact_match_accuracy(pred_seqs: List[str], gold_seqs: List[str]) -> float:
    """
    Exact match accuracy at word level.
    """
    assert len(pred_seqs) == len(gold_seqs)
    correct = sum(p == g for p, g in zip(pred_seqs, gold_seqs))
    return correct / len(gold_seqs) if gold_seqs else 0.0
