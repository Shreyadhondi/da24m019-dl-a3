from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"
UNK = "<UNK>"

SPECIAL_TOKENS = [PAD, SOS, EOS, UNK]


@dataclass
class Vocab:
    stoi: Dict[str, int]  # string to index
    itos: List[str]       # index to string

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def sos_id(self) -> int:
        return self.stoi[SOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    def encode_chars(self, text: str) -> List[int]:
        """Encode a string into a list of character ids."""
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode_chars(self, ids: Sequence[int], stop_at_eos: bool = True) -> str:
        """Decode character ids back to a string."""
        out_chars: List[str] = []
        for idx in ids:
            token = self.itos[idx]
            if stop_at_eos and token == EOS:
                break
            if token in SPECIAL_TOKENS:
                continue
            out_chars.append(token)
        return "".join(out_chars)


def build_char_vocab(texts: Sequence[str]) -> Vocab:
    """
    Build a character-level vocabulary from a list of strings.
    Adds SPECIAL_TOKENS at the beginning.
    """
    unique_chars = set()
    for t in texts:
        unique_chars.update(list(t))

    # fixed order: special tokens first, then sorted chars for reproducibility
    itos = SPECIAL_TOKENS + sorted(unique_chars)
    stoi = {ch: i for i, ch in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)


def add_sos_eos(ids: List[int], sos_id: int, eos_id: int) -> List[int]:
    """Add <SOS> at start and <EOS> at end."""
    return [sos_id] + ids + [eos_id]
