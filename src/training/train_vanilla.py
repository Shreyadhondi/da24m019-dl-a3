from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb

from src.data.load_dakshina import read_lexicon_tsv
from src.data.vocab import build_char_vocab
from src.data.dataset import DakshinaSeq2SeqDataset, collate_fn
from src.models.vanilla_seq2seq import Encoder, Decoder, VanillaSeq2Seq
from src.training.metrics import token_accuracy


# -------------------------------
# Small helpers
# -------------------------------
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ids_to_string(ids: List[int], itos: List[str], eos_id: int, skip_ids: set[int]) -> str:
    chars: List[str] = []
    for tid in ids:
        if tid == eos_id:
            break
        if tid in skip_ids:
            continue
        if 0 <= tid < len(itos):
            chars.append(itos[tid])
    return "".join(chars)


RNNState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


def _adapt_state_num_layers(state: RNNState, target_layers: int) -> RNNState:
    """
    Adapt encoder final state to match decoder num_layers.
    Works for GRU/RNN: Tensor [L, B, H]
    Works for LSTM: (h, c) each [L, B, H]
    Strategy:
      - If encoder layers > decoder layers: take last target_layers
      - If encoder layers < decoder layers: repeat last layer to expand
    """

    def adapt_tensor(h: torch.Tensor) -> torch.Tensor:
        # h: [L, B, H]
        L = h.size(0)
        if L == target_layers:
            return h
        if L > target_layers:
            return h[-target_layers:, ...]
        # L < target_layers: repeat last layer
        repeat = target_layers - L
        last = h[-1:, ...].repeat(repeat, 1, 1)
        return torch.cat([h, last], dim=0)

    if isinstance(state, tuple):
        h, c = state
        return adapt_tensor(h), adapt_tensor(c)
    else:
        return adapt_tensor(state)


@torch.no_grad()
def greedy_decode_batch(
    model: VanillaSeq2Seq,
    src: torch.Tensor,
    src_lens: torch.Tensor,
    tgt_vocab_itos: List[str],
    tgt_sos_id: int,
    tgt_eos_id: int,
    tgt_pad_id: int,
    max_len: int = 40,
) -> List[str]:
    """
    Greedy decoding for a batch.
    Returns list of predicted strings (one per example).
    """
    model.eval()

    # encode
    _, enc_state = model.encoder(src, src_lens)

    # adapt encoder state to decoder num_layers (fixes sweep crashes when enc_layers != dec_layers)
    dec_layers = model.decoder.num_layers  # attribute in your Decoder class (should exist)
    dec_state = _adapt_state_num_layers(enc_state, dec_layers)

    B = src.size(0)
    inp = torch.full((B,), tgt_sos_id, dtype=torch.long, device=src.device)

    preds: List[List[int]] = [[] for _ in range(B)]
    finished = torch.zeros((B,), dtype=torch.bool, device=src.device)

    skip_ids = {tgt_sos_id, tgt_pad_id}

    for _ in range(max_len):
        logits, dec_state = model.decoder.forward_step(inp, dec_state)
        next_ids = logits.argmax(dim=1)  # [B]
        inp = next_ids

        for i in range(B):
            if not finished[i]:
                preds[i].append(int(next_ids[i].item()))
                if int(next_ids[i].item()) == tgt_eos_id:
                    finished[i] = True

        if bool(finished.all()):
            break

    pred_strs = [ids_to_string(seq, tgt_vocab_itos, tgt_eos_id, skip_ids=skip_ids) for seq in preds]
    return pred_strs


def exact_match_accuracy(preds: List[str], gold: List[str]) -> float:
    assert len(preds) == len(gold)
    if len(gold) == 0:
        return 0.0
    correct = sum(p == g for p, g in zip(preds, gold))
    return correct / len(gold)


def build_loaders_and_vocabs(
    data_dir: Path, lang: str, batch_size: int
) -> Tuple[DataLoader, DataLoader, Dict, Dict]:
    """
    Returns:
      train_loader, dev_loader, src_vocab, tgt_vocab
    """
    train_path = data_dir / lang / "lexicons" / f"{lang}.translit.sampled.train.tsv"
    dev_path = data_dir / lang / "lexicons" / f"{lang}.translit.sampled.dev.tsv"

    train_split = read_lexicon_tsv(train_path)
    dev_split = read_lexicon_tsv(dev_path)

    # Build vocabs from TRAIN only
    train_romans = [x for x, _ in train_split.pairs]
    train_natives = [y for _, y in train_split.pairs]

    src_vocab = build_char_vocab(train_romans)
    tgt_vocab = build_char_vocab(train_natives)

    train_ds = DakshinaSeq2SeqDataset(train_split.pairs, src_vocab, tgt_vocab)
    dev_ds = DakshinaSeq2SeqDataset(dev_split.pairs, src_vocab, tgt_vocab)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, src_vocab.pad_id, tgt_vocab.pad_id),
        num_workers=0,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, src_vocab.pad_id, tgt_vocab.pad_id),
        num_workers=0,
    )

    return train_loader, dev_loader, src_vocab, tgt_vocab


def build_test_loader(
    data_dir: Path, lang: str, batch_size: int, src_vocab: Dict, tgt_vocab: Dict
) -> DataLoader:
    test_path = data_dir / lang / "lexicons" / f"{lang}.translit.sampled.test.tsv"
    test_split = read_lexicon_tsv(test_path)
    test_ds = DakshinaSeq2SeqDataset(test_split.pairs, src_vocab, tgt_vocab)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, src_vocab.pad_id, tgt_vocab.pad_id),
        num_workers=0,
    )
    return test_loader


def build_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    pad_id: int,
    emb_size: int,
    hidden_size: int,
    enc_layers: int,
    dec_layers: int,
    cell_type: str,
    dropout: float,
) -> VanillaSeq2Seq:
    encoder = Encoder(
        vocab_size=src_vocab_size,
        emb_size=emb_size,
        hidden_size=hidden_size,
        num_layers=enc_layers,
        cell_type=cell_type,  # type: ignore[arg-type]
        dropout=dropout,
    )
    decoder = Decoder(
        vocab_size=tgt_vocab_size,
        emb_size=emb_size,
        hidden_size=hidden_size,
        num_layers=dec_layers,
        cell_type=cell_type,  # type: ignore[arg-type]
        dropout=dropout,
    )
    return VanillaSeq2Seq(encoder=encoder, decoder=decoder, pad_id=pad_id)


def train_one_epoch(
    model: VanillaSeq2Seq,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    teacher_forcing_ratio: float,
    tgt_pad_id: int,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for src, tgt, src_lens, tgt_lens in loader:
        src = src.to(device)
        tgt = tgt.to(device)
        src_lens = src_lens.to(device)

        optimizer.zero_grad()

        logits = model(src, src_lens, tgt, teacher_forcing_ratio=teacher_forcing_ratio)
        targets = tgt[:, 1:]

        B, Tm1, V = logits.shape
        loss = criterion(logits.reshape(B * Tm1, V), targets.reshape(B * Tm1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        acc = token_accuracy(logits, targets, pad_id=tgt_pad_id)

        total_loss += float(loss.item())
        total_acc += float(acc)
        n_batches += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: VanillaSeq2Seq,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tgt_vocab_itos: List[str],
    tgt_sos_id: int,
    tgt_eos_id: int,
    tgt_pad_id: int,
    max_len: int,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_tok_acc = 0.0
    n_batches = 0

    all_pred: List[str] = []
    all_gold: List[str] = []

    skip_ids = {tgt_sos_id, tgt_pad_id}

    for src, tgt, src_lens, tgt_lens in loader:
        src = src.to(device)
        tgt = tgt.to(device)
        src_lens = src_lens.to(device)

        logits = model(src, src_lens, tgt, teacher_forcing_ratio=1.0)
        targets = tgt[:, 1:]
        B, Tm1, V = logits.shape
        loss = criterion(logits.reshape(B * Tm1, V), targets.reshape(B * Tm1))

        tok_acc = token_accuracy(logits, targets, pad_id=tgt_pad_id)

        total_loss += float(loss.item())
        total_tok_acc += float(tok_acc)
        n_batches += 1

        pred_strs = greedy_decode_batch(
            model,
            src,
            src_lens,
            tgt_vocab_itos=tgt_vocab_itos,
            tgt_sos_id=tgt_sos_id,
            tgt_eos_id=tgt_eos_id,
            tgt_pad_id=tgt_pad_id,
            max_len=max_len,
        )

        gold_ids = tgt.tolist()
        for seq in gold_ids:
            gold_str = ids_to_string(seq, tgt_vocab_itos, eos_id=tgt_eos_id, skip_ids=skip_ids)
            all_gold.append(gold_str)

        all_pred.extend(pred_strs)

    dev_exact = exact_match_accuracy(all_pred, all_gold)
    return total_loss / max(n_batches, 1), total_tok_acc / max(n_batches, 1), dev_exact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data folder (top-level).")
    parser.add_argument("--lang", type=str, default="te", help="Language code (e.g., te, hi, ta).")

    parser.add_argument("--project", type=str, default="da6401-a3", help="W&B project name.")
    parser.add_argument("--run_name", type=str, default=None, help="Optional W&B run name.")
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging.")

    # NEW: choose checkpoint filename so global best doesn't get overwritten
    parser.add_argument("--ckpt_name", type=str, default="best_vanilla.pt", help="Checkpoint filename inside checkpoints/")

    # NEW: optionally evaluate on test after training
    parser.add_argument("--eval_test", action="store_true", help="Evaluate on test set after training finishes.")

    # defaults (can be overridden by wandb.config in sweeps)
    parser.add_argument("--emb_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--enc_layers", type=int, default=1)
    parser.add_argument("--dec_layers", type=int, default=1)
    parser.add_argument("--cell_type", type=str, default="gru", choices=["rnn", "gru", "lstm"])
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--teacher_forcing", type=float, default=1.0)
    parser.add_argument("--max_decode_len", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------------
    # W&B init (optional)
    # -------------------------------
    if args.use_wandb:
        wandb.init(project=args.project if args.project else None, name=args.run_name)
        cfg = wandb.config

        emb_size = int(getattr(cfg, "emb_size", args.emb_size))
        hidden_size = int(getattr(cfg, "hidden_size", args.hidden_size))
        enc_layers = int(getattr(cfg, "enc_layers", args.enc_layers))
        dec_layers = int(getattr(cfg, "dec_layers", args.dec_layers))
        cell_type = str(getattr(cfg, "cell_type", args.cell_type))
        dropout = float(getattr(cfg, "dropout", args.dropout))

        batch_size = int(getattr(cfg, "batch_size", args.batch_size))
        lr = float(getattr(cfg, "lr", args.lr))
        epochs = int(getattr(cfg, "epochs", args.epochs))
        teacher_forcing = float(getattr(cfg, "teacher_forcing", args.teacher_forcing))
        max_decode_len = int(getattr(cfg, "max_decode_len", args.max_decode_len))

        # Nice run name (works in sweeps)
        if wandb.run is not None:
            run_name = (
                f"{cell_type}_E{emb_size}_H{hidden_size}"
                f"_enc{enc_layers}_dec{dec_layers}"
                f"_do{dropout}_lr{lr}"
            )
            wandb.run.name = run_name
    else:
        emb_size = args.emb_size
        hidden_size = args.hidden_size
        enc_layers = args.enc_layers
        dec_layers = args.dec_layers
        cell_type = args.cell_type
        dropout = args.dropout

        batch_size = args.batch_size
        lr = args.lr
        epochs = args.epochs
        teacher_forcing = args.teacher_forcing
        max_decode_len = args.max_decode_len

    print("\nHyperparameters:")
    print(f"  emb_size={emb_size}, hidden_size={hidden_size}, cell={cell_type}")
    print(f"  enc_layers={enc_layers}, dec_layers={dec_layers}, dropout={dropout}")
    print(f"  batch_size={batch_size}, lr={lr}, epochs={epochs}, teacher_forcing={teacher_forcing}\n")

    # -------------------------------
    # Data
    # -------------------------------
    data_dir = Path(args.data_dir)
    train_loader, dev_loader, src_vocab, tgt_vocab = build_loaders_and_vocabs(
        data_dir=data_dir,
        lang=args.lang,
        batch_size=batch_size,
    )

    # -------------------------------
    # Model
    # -------------------------------
    model = build_model(
        src_vocab_size=len(src_vocab.itos),
        tgt_vocab_size=len(tgt_vocab.itos),
        pad_id=tgt_vocab.pad_id,
        emb_size=emb_size,
        hidden_size=hidden_size,
        enc_layers=enc_layers,
        dec_layers=dec_layers,
        cell_type=cell_type,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_id)

    # -------------------------------
    # Training loop
    # -------------------------------
    best_dev_exact = 0.0
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / args.ckpt_name

    for epoch in range(1, epochs + 1):
        train_loss, train_tok_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            teacher_forcing_ratio=teacher_forcing,
            tgt_pad_id=tgt_vocab.pad_id,
        )

        dev_loss, dev_tok_acc, dev_exact = evaluate(
            model=model,
            loader=dev_loader,
            criterion=criterion,
            device=device,
            tgt_vocab_itos=tgt_vocab.itos,
            tgt_sos_id=tgt_vocab.sos_id,
            tgt_eos_id=tgt_vocab.eos_id,
            tgt_pad_id=tgt_vocab.pad_id,
            max_len=max_decode_len,
        )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_tok_acc={train_tok_acc:.4f} | "
            f"dev_loss={dev_loss:.4f} dev_tok_acc={dev_tok_acc:.4f} dev_exact={dev_exact:.4f}"
        )

        if args.use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_token_acc": train_tok_acc,
                    "dev_loss": dev_loss,
                    "dev_token_acc": dev_tok_acc,
                    "dev_exact_match": dev_exact,
                    "best_dev_exact_match": best_dev_exact,  # so you can sort easily
                }
            )

        # Save best checkpoint (per run)
        if dev_exact > best_dev_exact:
            best_dev_exact = dev_exact
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "src_vocab_itos": src_vocab.itos,
                    "tgt_vocab_itos": tgt_vocab.itos,
                    "config": {
                        "emb_size": emb_size,
                        "hidden_size": hidden_size,
                        "enc_layers": enc_layers,
                        "dec_layers": dec_layers,
                        "cell_type": cell_type,
                        "dropout": dropout,
                        "lr": lr,
                        "batch_size": batch_size,
                        "teacher_forcing": teacher_forcing,
                        "max_decode_len": max_decode_len,
                        "seed": args.seed,
                    },
                },
                ckpt_path,
            )

    print("\nBest dev exact match:", best_dev_exact)

    # -------------------------------
    # Optional: evaluate on test
    # -------------------------------
    if args.eval_test:
        test_loader = build_test_loader(data_dir, args.lang, batch_size, src_vocab, tgt_vocab)
        test_loss, test_tok_acc, test_exact = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            tgt_vocab_itos=tgt_vocab.itos,
            tgt_sos_id=tgt_vocab.sos_id,
            tgt_eos_id=tgt_vocab.eos_id,
            tgt_pad_id=tgt_vocab.pad_id,
            max_len=max_decode_len,
        )
        print(
            f"\nTEST | loss={test_loss:.4f} tok_acc={test_tok_acc:.4f} exact={test_exact:.4f}"
        )
        if args.use_wandb:
            wandb.log({"test_loss": test_loss, "test_token_acc": test_tok_acc, "test_exact_match": test_exact})

    if args.use_wandb:
        wandb.log({"best_dev_exact_match": best_dev_exact})
        wandb.finish()


if __name__ == "__main__":
    main()
