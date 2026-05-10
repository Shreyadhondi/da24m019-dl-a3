from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


RNNState = torch.Tensor
LSTMState = Tuple[torch.Tensor, torch.Tensor]
AnyState = torch.Tensor | Tuple[torch.Tensor, torch.Tensor]


def _make_rnn(
    cell_type: str,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
) -> nn.Module:
    cell_type = cell_type.lower()
    rnn_dropout = dropout if num_layers > 1 else 0.0

    if cell_type == "rnn":
        return nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=rnn_dropout,
            batch_first=True,
        )
    if cell_type == "gru":
        return nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=rnn_dropout,
            batch_first=True,
        )
    if cell_type == "lstm":
        return nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=rnn_dropout,
            batch_first=True,
        )
    raise ValueError(f"Unknown cell_type: {cell_type}")


def _is_lstm(state: AnyState) -> bool:
    return isinstance(state, tuple)


def _get_last_layer_hidden(state: AnyState) -> torch.Tensor:
    # returns [B, H] hidden from last layer
    if _is_lstm(state):
        h, _c = state
        return h[-1]  # [L, B, H] -> [B, H]
    return state[-1]  # [L, B, H] -> [B, H]


def _slice_layers(state: AnyState, num_layers: int) -> AnyState:
    # keep only top num_layers (normally 1 for our attention model)
    if _is_lstm(state):
        h, c = state
        return (h[:num_layers], c[:num_layers])
    return state[:num_layers]


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_size: int,
        hidden_size: int,
        num_layers: int = 1,
        cell_type: str = "gru",
        dropout: float = 0.2,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=pad_id)
        self.rnn = _make_rnn(cell_type, emb_size, hidden_size, num_layers, dropout)
        self.cell_type = cell_type.lower()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(
        self, src: torch.Tensor, src_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, AnyState]:
        # src: [B, T]
        emb = self.emb(src)  # [B, T, E]

        # pack for speed + correct handling
        packed = pack_padded_sequence(
            emb, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, state = self.rnn(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)  # [B, T, H]

        return out, state


class DotAttention(nn.Module):
    """
    Dot-product attention:
      score_t(s) = dot(dec_hidden_t, enc_out_s)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        dec_hidden: torch.Tensor,     # [B, H]
        enc_out: torch.Tensor,        # [B, Tsrc, H]
        src_lens: torch.Tensor,       # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # scores: [B, Tsrc]
        scores = torch.bmm(enc_out, dec_hidden.unsqueeze(2)).squeeze(2)

        # mask padding positions
        B, Tsrc = scores.shape
        device = scores.device
        idxs = torch.arange(Tsrc, device=device).unsqueeze(0).expand(B, Tsrc)
        mask = idxs >= src_lens.unsqueeze(1)  # True where padded
        scores = scores.masked_fill(mask, -1e9)

        attn = torch.softmax(scores, dim=1)  # [B, Tsrc]
        context = torch.bmm(attn.unsqueeze(1), enc_out).squeeze(1)  # [B, H]
        return context, attn


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_size: int,
        hidden_size: int,
        num_layers: int = 1,
        cell_type: str = "gru",
        dropout: float = 0.2,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=pad_id)
        self.attn = DotAttention()

        # We feed [emb ; context] to the RNN
        self.rnn = _make_rnn(
            cell_type=cell_type,
            input_size=emb_size + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.cell_type = cell_type.lower()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Output layer from hidden -> vocab
        self.proj = nn.Linear(hidden_size, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward_step(
        self,
        inp_tokens: torch.Tensor,     # [B]
        state: AnyState,              # (h,c) or h
        enc_out: torch.Tensor,        # [B, Tsrc, H]
        src_lens: torch.Tensor,       # [B]
    ) -> Tuple[torch.Tensor, AnyState, torch.Tensor]:
        # emb: [B, 1, E]
        emb = self.emb(inp_tokens).unsqueeze(1)
        emb = self.dropout(emb)

        dec_hidden = _get_last_layer_hidden(state)  # [B, H]
        context, attn = self.attn(dec_hidden, enc_out, src_lens)  # [B,H], [B,Tsrc]

        # concat: [B, 1, E+H]
        rnn_in = torch.cat([emb, context.unsqueeze(1)], dim=2)
        out, new_state = self.rnn(rnn_in, state)  # out: [B,1,H]
        out = out.squeeze(1)  # [B,H]
        logits = self.proj(out)  # [B,V]
        return logits, new_state, attn


class AttnSeq2Seq(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: AttentionDecoder,
        pad_id: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_id = pad_id

    def forward(
        self,
        src: torch.Tensor,
        src_lens: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
        return_attn: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns logits: [B, Ttgt-1, V]
        If return_attn=True: also returns attn_weights [B, Ttgt-1, Tsrc]
        """
        enc_out, enc_state = self.encoder(src, src_lens)

        # IMPORTANT: keep decoder layers consistent (we use 1 layer in attention)
        enc_state = _slice_layers(enc_state, self.decoder.num_layers)

        B, Ttgt = tgt.shape
        logits_all: List[torch.Tensor] = []
        attn_all: List[torch.Tensor] = []

        # first input to decoder is <sos> (tgt[:,0])
        inp = tgt[:, 0]  # [B]

        state = enc_state

        for t in range(1, Ttgt):
            logits, state, attn = self.decoder.forward_step(inp, state, enc_out, src_lens)
            logits_all.append(logits.unsqueeze(1))  # [B,1,V]
            if return_attn:
                attn_all.append(attn.unsqueeze(1))  # [B,1,Tsrc]

            use_teacher = (torch.rand(1, device=src.device).item() < teacher_forcing_ratio)
            if use_teacher:
                inp = tgt[:, t]
            else:
                inp = logits.argmax(dim=1)

        logits_seq = torch.cat(logits_all, dim=1)  # [B,Ttgt-1,V]
        if not return_attn:
            return logits_seq

        attn_seq = torch.cat(attn_all, dim=1)  # [B,Ttgt-1,Tsrc]
        return logits_seq, attn_seq
