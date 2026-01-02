from __future__ import annotations

from typing import Literal, Optional, Tuple, Union

import torch
import torch.nn as nn

CellType = Literal["rnn", "gru", "lstm"]
RNNState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]  # h or (h, c)


def make_rnn(
    cell_type: CellType,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
) -> nn.Module:
    """
    Create a PyTorch RNN/GRU/LSTM module.
    Note: PyTorch applies dropout ONLY between stacked layers (num_layers > 1).
    """
    do = dropout if num_layers > 1 else 0.0

    if cell_type == "rnn":
        return nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=do)
    if cell_type == "gru":
        return nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=do)
    if cell_type == "lstm":
        return nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=do)

    raise ValueError(f"Unknown cell type: {cell_type}")


class Encoder(nn.Module):
    """
    Encoder: Embedding + RNN/GRU/LSTM over source character ids.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_size: int,
        hidden_size: int,
        num_layers: int,
        cell_type: CellType,
        dropout: float,
    ):
        super().__init__()
        self.cell_type = cell_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.rnn = make_rnn(cell_type, emb_size, hidden_size, num_layers, dropout)

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor) -> Tuple[torch.Tensor, RNNState]:
        """
        Args:
            src: [B, T_src] (padded)
            src_lens: [B] lengths (before padding)
        Returns:
            outputs: [B, T_src, H]
            state: h  or  (h, c) for LSTM
        """
        emb = self.embedding(src)  # [B, T_src, E]

        packed = nn.utils.rnn.pack_padded_sequence(
            emb, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, state = self.rnn(packed)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return outputs, state


class Decoder(nn.Module):
    """
    Decoder: Embedding + RNN/GRU/LSTM, generates one token at a time.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_size: int,
        hidden_size: int,
        num_layers: int,
        cell_type: CellType,
        dropout: float,
    ):
        super().__init__()
        self.cell_type = cell_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.rnn = make_rnn(cell_type, emb_size, hidden_size, num_layers, dropout)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward_step(self, input_token: torch.Tensor, state: RNNState) -> Tuple[torch.Tensor, RNNState]:
        """
        One decoding step.
        Args:
            input_token: [B] token ids
            state: previous hidden state
        Returns:
            logits: [B, V]
            new_state: updated state
        """
        emb = self.embedding(input_token).unsqueeze(1)  # [B, 1, E]
        out, new_state = self.rnn(emb, state)          # out: [B, 1, H]
        logits = self.out(out.squeeze(1))              # [B, V]
        return logits, new_state


def _adapt_state_for_decoder(
    enc_state: RNNState,
    enc_layers: int,
    dec_layers: int,
    cell_type: CellType,
) -> RNNState:
    """
    If encoder and decoder layer counts differ, adapt by:
    - if dec_layers <= enc_layers: take top dec_layers
    - if dec_layers > enc_layers: repeat last layer to match
    """
    def adapt_h(h: torch.Tensor) -> torch.Tensor:
        # h: [L, B, H]
        if dec_layers == enc_layers:
            return h
        if dec_layers < enc_layers:
            return h[-dec_layers:]
        # dec_layers > enc_layers
        last = h[-1:].repeat(dec_layers - enc_layers, 1, 1)
        return torch.cat([h, last], dim=0)

    if cell_type == "lstm":
        h, c = enc_state  # type: ignore[misc]
        return adapt_h(h), adapt_h(c)

    # RNN/GRU
    h = enc_state  # type: ignore[assignment]
    return adapt_h(h)


class VanillaSeq2Seq(nn.Module):
    """
    Vanilla seq2seq: encoder final state initializes decoder.
    No attention.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, pad_id: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_id = pad_id

        # sanity: encoder/decoder cell types must match for state transfer
        if encoder.cell_type != decoder.cell_type:
            raise ValueError("Encoder and Decoder cell_type must be the same in vanilla seq2seq.")

    def forward(
        self,
        src: torch.Tensor,
        src_lens: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        """
        Training forward with teacher forcing.
        Args:
            src: [B, T_src]
            src_lens: [B]
            tgt: [B, T_tgt] includes <SOS> ... <EOS> ... <PAD>
            teacher_forcing_ratio: float in [0,1]
        Returns:
            logits: [B, T_tgt-1, V] predicting tgt[:, 1:] from tgt[:, :-1]
        """
        _, enc_state = self.encoder(src, src_lens)

        # adapt state if encoder/decoder have different layer counts
        dec_state = _adapt_state_for_decoder(
            enc_state,
            enc_layers=self.encoder.num_layers,
            dec_layers=self.decoder.num_layers,
            cell_type=self.encoder.cell_type,
        )

        B, T_tgt = tgt.shape
        inp = tgt[:, 0]  # <SOS>
        logits_steps = []

        # Use a device-safe random draw for teacher forcing
        for t in range(1, T_tgt):
            logits, dec_state = self.decoder.forward_step(inp, dec_state)
            logits_steps.append(logits.unsqueeze(1))  # [B,1,V]

            if teacher_forcing_ratio >= 1.0:
                inp = tgt[:, t]
            elif teacher_forcing_ratio <= 0.0:
                inp = logits.argmax(dim=1)
            else:
                # draw on the same device
                use_teacher = (torch.rand((), device=tgt.device) < teacher_forcing_ratio).item()
                inp = tgt[:, t] if use_teacher else logits.argmax(dim=1)

        return torch.cat(logits_steps, dim=1)
