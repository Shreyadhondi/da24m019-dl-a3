from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from src.data.vocab import Vocab, add_sos_eos


@dataclass
class Seq2SeqExample:
    src_ids: List[int]  # roman ids (with SOS/EOS)
    tgt_ids: List[int]  # telugu ids (with SOS/EOS)


class DakshinaSeq2SeqDataset(Dataset):
    """
    Dataset that converts (roman, native) pairs into (src_ids, tgt_ids).
    """

    def __init__(self, pairs: List[Tuple[str, str]], src_vocab: Vocab, tgt_vocab: Vocab):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Seq2SeqExample:
        roman, native = self.pairs[idx]

        src = add_sos_eos(
            self.src_vocab.encode_chars(roman),
            self.src_vocab.sos_id,
            self.src_vocab.eos_id,
        )
        tgt = add_sos_eos(
            self.tgt_vocab.encode_chars(native),
            self.tgt_vocab.sos_id,
            self.tgt_vocab.eos_id,
        )
        return Seq2SeqExample(src_ids=src, tgt_ids=tgt)


def pad_1d(seqs: List[List[int]], pad_id: int) -> torch.Tensor:
    """
    Pads list of variable-length sequences into a tensor [B, T].
    """
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def collate_fn(batch: List[Seq2SeqExample], src_pad_id: int, tgt_pad_id: int):
    """
    Creates a padded batch.

    Returns:
        src: [B, Tsrc]
        tgt: [B, Ttgt]
        src_lens: [B]
        tgt_lens: [B]
    """
    src_seqs = [ex.src_ids for ex in batch]
    tgt_seqs = [ex.tgt_ids for ex in batch]

    src_lens = torch.tensor([len(s) for s in src_seqs], dtype=torch.long)
    tgt_lens = torch.tensor([len(t) for t in tgt_seqs], dtype=torch.long)

    src = pad_1d(src_seqs, src_pad_id)
    tgt = pad_1d(tgt_seqs, tgt_pad_id)

    return src, tgt, src_lens, tgt_lens
