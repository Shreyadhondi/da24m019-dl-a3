from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from src.data.load_dakshina import read_lexicon_tsv
from src.data.vocab import build_char_vocab
from src.data.dataset import DakshinaSeq2SeqDataset, collate_fn
from src.models.attention import Encoder, AttentionDecoder, AttnSeq2Seq
from src.training.metrics import token_accuracy


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


def exact_match_accuracy(preds: List[str], golds: List[str]) -> float:
    if not golds:
        return 0.0
    return sum(p == g for p, g in zip(preds, golds)) / len(golds)


def build_loaders_and_vocabs(
    data_dir: Path, lang: str, batch_size: int
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict, Dict]:
    train_path = data_dir / lang / "lexicons" / f"{lang}.translit.sampled.train.tsv"
    dev_path = data_dir / lang / "lexicons" / f"{lang}.translit.sampled.dev.tsv"
    test_path = data_dir / lang / "lexicons" / f"{lang}.translit.sampled.test.tsv"

    train_split = read_lexicon_tsv(train_path)
    dev_split = read_lexicon_tsv(dev_path)
    test_split = read_lexicon_tsv(test_path)

    train_romans = [x for x, _ in train_split.pairs]
    train_natives = [y for _, y in train_split.pairs]

    src_vocab = build_char_vocab(train_romans)
    tgt_vocab = build_char_vocab(train_natives)

    train_ds = DakshinaSeq2SeqDataset(train_split.pairs, src_vocab, tgt_vocab)
    dev_ds = DakshinaSeq2SeqDataset(dev_split.pairs, src_vocab, tgt_vocab)
    test_ds = DakshinaSeq2SeqDataset(test_split.pairs, src_vocab, tgt_vocab)

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
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, src_vocab.pad_id, tgt_vocab.pad_id),
        num_workers=0,
    )

    return train_loader, dev_loader, test_loader, src_vocab, tgt_vocab


@torch.no_grad()
def greedy_decode_ids(
    model: AttnSeq2Seq,
    src: torch.Tensor,
    src_lens: torch.Tensor,
    tgt_sos_id: int,
    tgt_eos_id: int,
    max_len: int,
) -> List[List[int]]:
    model.eval()

    enc_out, enc_state = model.encoder(src, src_lens)

    if isinstance(enc_state, torch.Tensor):
        state = enc_state[: model.decoder.num_layers]
    else:
        state = (
            enc_state[0][: model.decoder.num_layers],
            enc_state[1][: model.decoder.num_layers],
        )

    batch_size = src.size(0)
    inp = torch.full((batch_size,), tgt_sos_id, dtype=torch.long, device=src.device)
    finished = torch.zeros((batch_size,), dtype=torch.bool, device=src.device)

    pred_ids: List[List[int]] = [[] for _ in range(batch_size)]

    for _ in range(max_len):
        logits, state, _attn = model.decoder.forward_step(inp, state, enc_out, src_lens)
        nxt = logits.argmax(dim=1)
        inp = nxt

        for i in range(batch_size):
            if not finished[i]:
                token_id = int(nxt[i].item())
                pred_ids[i].append(token_id)
                if token_id == tgt_eos_id:
                    finished[i] = True

        if bool(finished.all()):
            break

    return pred_ids


def train_one_epoch(
    model: AttnSeq2Seq,
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

    for src, tgt, src_lens, _tgt_lens in loader:
        src = src.to(device)
        tgt = tgt.to(device)
        src_lens = src_lens.to(device)

        optimizer.zero_grad()

        logits = model(src, src_lens, tgt, teacher_forcing_ratio=teacher_forcing_ratio)
        targets = tgt[:, 1:]

        batch_size, seq_len, vocab_size = logits.shape
        loss = criterion(
            logits.reshape(batch_size * seq_len, vocab_size),
            targets.reshape(batch_size * seq_len),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        acc = token_accuracy(logits, targets, pad_id=tgt_pad_id)

        total_loss += float(loss.item())
        total_acc += float(acc)
        n_batches += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


@torch.no_grad()
def evaluate_dev(
    model: AttnSeq2Seq,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tgt_vocab_itos: List[str],
    tgt_sos_id: int,
    tgt_eos_id: int,
    tgt_pad_id: int,
    max_decode_len: int,
) -> Tuple[float, float, float]:
    model.eval()

    total_loss = 0.0
    total_tok_acc = 0.0
    n_batches = 0

    all_preds: List[str] = []
    all_golds: List[str] = []
    skip_ids = {tgt_sos_id, tgt_pad_id}

    for src, tgt, src_lens, _tgt_lens in loader:
        src = src.to(device)
        tgt = tgt.to(device)
        src_lens = src_lens.to(device)

        logits = model(src, src_lens, tgt, teacher_forcing_ratio=1.0)
        targets = tgt[:, 1:]

        batch_size, seq_len, vocab_size = logits.shape
        loss = criterion(
            logits.reshape(batch_size * seq_len, vocab_size),
            targets.reshape(batch_size * seq_len),
        )

        tok_acc = token_accuracy(logits, targets, pad_id=tgt_pad_id)

        pred_ids_batch = greedy_decode_ids(
            model=model,
            src=src,
            src_lens=src_lens,
            tgt_sos_id=tgt_sos_id,
            tgt_eos_id=tgt_eos_id,
            max_len=max_decode_len,
        )

        for pred_ids in pred_ids_batch:
            pred_str = ids_to_string(
                pred_ids,
                itos=tgt_vocab_itos,
                eos_id=tgt_eos_id,
                skip_ids=skip_ids,
            )
            all_preds.append(pred_str)

        for gold_ids in tgt.detach().cpu().tolist():
            gold_str = ids_to_string(
                gold_ids,
                itos=tgt_vocab_itos,
                eos_id=tgt_eos_id,
                skip_ids=skip_ids,
            )
            all_golds.append(gold_str)

        total_loss += float(loss.item())
        total_tok_acc += float(tok_acc)
        n_batches += 1

    exact = exact_match_accuracy(all_preds, all_golds)

    return (
        total_loss / max(n_batches, 1),
        total_tok_acc / max(n_batches, 1),
        exact,
    )


def save_checkpoint(
    ckpt_path: Path,
    model: AttnSeq2Seq,
    src_vocab,
    tgt_vocab,
    config: Dict,
    dev_exact: float,
) -> None:
    ckpt_path.parent.mkdir(exist_ok=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "src_vocab_itos": src_vocab.itos,
            "tgt_vocab_itos": tgt_vocab.itos,
            "config": config,
            "dev_exact": dev_exact,
        },
        ckpt_path,
    )


def maybe_save_global_best(
    model: AttnSeq2Seq,
    src_vocab,
    tgt_vocab,
    config: Dict,
    dev_exact: float,
    ckpt_dir: Path,
) -> None:
    """
    Saves global_best_attention.pt across all sweep runs on this machine.

    It uses a JSON metadata file to remember the best score so far.
    Works well when W&B agent is running runs sequentially.
    """
    ckpt_dir.mkdir(exist_ok=True)

    global_ckpt_path = ckpt_dir / "global_best_attention.pt"
    meta_path = ckpt_dir / "global_best_attention_meta.json"

    old_best = -1.0
    if meta_path.exists():
        try:
            old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            old_best = float(old_meta.get("best_dev_exact", -1.0))
        except Exception:
            old_best = -1.0

    if dev_exact > old_best:
        save_checkpoint(
            ckpt_path=global_ckpt_path,
            model=model,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            config=config,
            dev_exact=dev_exact,
        )

        meta = {
            "best_dev_exact": dev_exact,
            "config": config,
            "wandb_run_name": wandb.run.name if wandb.run is not None else None,
            "wandb_run_id": wandb.run.id if wandb.run is not None else None,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print(f"UPDATED GLOBAL BEST: {dev_exact:.6f}")
        print(f"Saved global best checkpoint: {global_ckpt_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--lang", type=str, default="te")

    parser.add_argument("--project", type=str, default="da24m019-dl-a3")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")

    parser.add_argument("--emb_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--cell_type", type=str, default="gru", choices=["rnn", "gru", "lstm"])
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--enc_layers", type=int, default=1)
    parser.add_argument("--dec_layers", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--teacher_forcing", type=float, default=1.0)
    parser.add_argument("--max_decode_len", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--ckpt_name", type=str, default="best_attention.pt")

    args = parser.parse_args()

    if args.enc_layers != 1 or args.dec_layers != 1:
        raise ValueError(
            "For attention part, use enc_layers=1 and dec_layers=1. "
            "The assignment allows a single-layer encoder and decoder."
        )

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if args.use_wandb:
        wandb.init(project=args.project, name=args.run_name)
        cfg = wandb.config

        emb_size = int(getattr(cfg, "emb_size", args.emb_size))
        hidden_size = int(getattr(cfg, "hidden_size", args.hidden_size))
        cell_type = str(getattr(cfg, "cell_type", args.cell_type))
        dropout = float(getattr(cfg, "dropout", args.dropout))

        batch_size = int(getattr(cfg, "batch_size", args.batch_size))
        lr = float(getattr(cfg, "lr", args.lr))
        epochs = int(getattr(cfg, "epochs", args.epochs))
        teacher_forcing = float(getattr(cfg, "teacher_forcing", args.teacher_forcing))
        max_decode_len = int(getattr(cfg, "max_decode_len", args.max_decode_len))

        if wandb.run is not None:
            wandb.run.name = (
                f"attention_{cell_type}"
                f"_E{emb_size}"
                f"_H{hidden_size}"
                f"_do{dropout}"
                f"_lr{lr}"
            )
    else:
        emb_size = args.emb_size
        hidden_size = args.hidden_size
        cell_type = args.cell_type
        dropout = args.dropout

        batch_size = args.batch_size
        lr = args.lr
        epochs = args.epochs
        teacher_forcing = args.teacher_forcing
        max_decode_len = args.max_decode_len

    print("\nHyperparameters (ATTENTION):")
    print(f"  emb_size={emb_size}, hidden_size={hidden_size}, cell={cell_type}")
    print(f"  enc_layers=1, dec_layers=1, dropout={dropout}")
    print(f"  batch_size={batch_size}, lr={lr}, epochs={epochs}, teacher_forcing={teacher_forcing}\n")

    train_loader, dev_loader, _test_loader, src_vocab, tgt_vocab = build_loaders_and_vocabs(
        data_dir=Path(args.data_dir),
        lang=args.lang,
        batch_size=batch_size,
    )

    encoder = Encoder(
        vocab_size=len(src_vocab.itos),
        emb_size=emb_size,
        hidden_size=hidden_size,
        num_layers=1,
        cell_type=cell_type,
        dropout=dropout,
        pad_id=src_vocab.pad_id,
    )

    decoder = AttentionDecoder(
        vocab_size=len(tgt_vocab.itos),
        emb_size=emb_size,
        hidden_size=hidden_size,
        num_layers=1,
        cell_type=cell_type,
        dropout=dropout,
        pad_id=tgt_vocab.pad_id,
    )

    model = AttnSeq2Seq(
        encoder=encoder,
        decoder=decoder,
        pad_id=tgt_vocab.pad_id,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_id)

    run_config = {
        "emb_size": emb_size,
        "hidden_size": hidden_size,
        "cell_type": cell_type,
        "dropout": dropout,
        "enc_layers": 1,
        "dec_layers": 1,
        "lang": args.lang,
        "lr": lr,
        "batch_size": batch_size,
        "teacher_forcing": teacher_forcing,
        "max_decode_len": max_decode_len,
        "seed": args.seed,
    }

    best_dev_exact = -1.0

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    run_ckpt_path = ckpt_dir / args.ckpt_name

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

        dev_loss, dev_tok_acc, dev_exact = evaluate_dev(
            model=model,
            loader=dev_loader,
            criterion=criterion,
            device=device,
            tgt_vocab_itos=tgt_vocab.itos,
            tgt_sos_id=tgt_vocab.sos_id,
            tgt_eos_id=tgt_vocab.eos_id,
            tgt_pad_id=tgt_vocab.pad_id,
            max_decode_len=max_decode_len,
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
                    "best_dev_exact_match": max(best_dev_exact, dev_exact),
                }
            )

        if dev_exact > best_dev_exact:
            best_dev_exact = dev_exact

            # best checkpoint inside this run
            save_checkpoint(
                ckpt_path=run_ckpt_path,
                model=model,
                src_vocab=src_vocab,
                tgt_vocab=tgt_vocab,
                config=run_config,
                dev_exact=best_dev_exact,
            )

            # global best checkpoint across sweep runs
            maybe_save_global_best(
                model=model,
                src_vocab=src_vocab,
                tgt_vocab=tgt_vocab,
                config=run_config,
                dev_exact=best_dev_exact,
                ckpt_dir=ckpt_dir,
            )

    print("\nBest dev exact match:", best_dev_exact)
    print("Saved run-best checkpoint:", run_ckpt_path)
    print("Global best checkpoint path:", ckpt_dir / "global_best_attention.pt")

    if args.use_wandb:
        wandb.log({"best_dev_exact_match": best_dev_exact})
        wandb.finish()


if __name__ == "__main__":
    main()