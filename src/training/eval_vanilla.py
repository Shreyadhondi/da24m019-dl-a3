from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.load_dakshina import read_lexicon_tsv
from src.data.vocab import Vocab, build_char_vocab
from src.data.dataset import DakshinaSeq2SeqDataset, collate_fn
from src.models.vanilla_seq2seq import Encoder, Decoder, VanillaSeq2Seq
from src.training.metrics import token_accuracy


# -------------------------------
# Helpers
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
    model.eval()

    _, enc_state = model.encoder(src, src_lens)

    # decoder expects state shaped for dec_layers, but encoder may have enc_layers
    dec_layers = model.decoder.num_layers  # type: ignore[attr-defined]

    def adapt_tensor_layers(x: torch.Tensor, target_layers: int) -> torch.Tensor:
        # x: [L, B, H]
        L = x.size(0)
        if L == target_layers:
            return x
        if L > target_layers:
            return x[:target_layers]

        # L < target_layers -> repeat last layer
        pad = x[-1:].repeat(target_layers - L, 1, 1)
        return torch.cat([x, pad], dim=0)

    def adapt_state(state):
        # GRU/RNN: Tensor [L, B, H]
        # LSTM: Tuple(h, c), each [L, B, H]
        if isinstance(state, tuple):
            h, c = state
            h = adapt_tensor_layers(h, dec_layers)
            c = adapt_tensor_layers(c, dec_layers)
            return (h, c)

        return adapt_tensor_layers(state, dec_layers)

    dec_state = adapt_state(enc_state)

    batch_size = src.size(0)
    inp = torch.full((batch_size,), tgt_sos_id, dtype=torch.long, device=src.device)

    preds: List[List[int]] = [[] for _ in range(batch_size)]
    finished = torch.zeros((batch_size,), dtype=torch.bool, device=src.device)

    skip_ids = {tgt_sos_id, tgt_pad_id}

    for _ in range(max_len):
        logits, dec_state = model.decoder.forward_step(inp, dec_state)
        next_ids = logits.argmax(dim=1)
        inp = next_ids

        for i in range(batch_size):
            if not finished[i]:
                tid = int(next_ids[i].item())
                preds[i].append(tid)

                if tid == tgt_eos_id:
                    finished[i] = True

        if bool(finished.all()):
            break

    return [
        ids_to_string(seq, tgt_vocab_itos, tgt_eos_id, skip_ids)
        for seq in preds
    ]


def exact_match_accuracy(preds: List[str], gold: List[str]) -> float:
    if len(gold) == 0:
        return 0.0
    return sum(p == g for p, g in zip(preds, gold)) / len(gold)


def build_vocabs_from_train(data_dir: Path, lang: str) -> Tuple[Vocab, Vocab]:
    train_path = data_dir / lang / "lexicons" / f"{lang}.translit.sampled.train.tsv"

    train_split = read_lexicon_tsv(train_path)

    train_romans = [x for x, _ in train_split.pairs]
    train_natives = [y for _, y in train_split.pairs]

    src_vocab = build_char_vocab(train_romans)
    tgt_vocab = build_char_vocab(train_natives)

    return src_vocab, tgt_vocab


def build_loader(
    data_dir: Path,
    lang: str,
    split: str,
    batch_size: int,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
) -> Tuple[DataLoader, List[tuple[str, str]]]:
    path = data_dir / lang / "lexicons" / f"{lang}.translit.sampled.{split}.tsv"

    split_data = read_lexicon_tsv(path)
    pairs = split_data.pairs

    dataset = DakshinaSeq2SeqDataset(pairs, src_vocab, tgt_vocab)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, src_vocab.pad_id, tgt_vocab.pad_id),
        num_workers=0,
    )

    return loader, pairs


@torch.no_grad()
def predict_all(
    model: VanillaSeq2Seq,
    loader: DataLoader,
    device: torch.device,
    tgt_vocab_itos: List[str],
    tgt_sos_id: int,
    tgt_eos_id: int,
    tgt_pad_id: int,
    max_len: int,
) -> List[str]:
    model.eval()

    all_preds: List[str] = []

    for src, _tgt, src_lens, _tgt_lens in loader:
        src = src.to(device)
        src_lens = src_lens.to(device)

        preds = greedy_decode_batch(
            model=model,
            src=src,
            src_lens=src_lens,
            tgt_vocab_itos=tgt_vocab_itos,
            tgt_sos_id=tgt_sos_id,
            tgt_eos_id=tgt_eos_id,
            tgt_pad_id=tgt_pad_id,
            max_len=max_len,
        )

        all_preds.extend(preds)

    return all_preds


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
) -> tuple[float, float, float]:
    model.eval()

    total_loss = 0.0
    total_tok_acc = 0.0
    n_batches = 0

    all_pred: List[str] = []
    all_gold: List[str] = []

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

        total_loss += float(loss.item())
        total_tok_acc += float(tok_acc)
        n_batches += 1

        pred_strs = greedy_decode_batch(
            model=model,
            src=src,
            src_lens=src_lens,
            tgt_vocab_itos=tgt_vocab_itos,
            tgt_sos_id=tgt_sos_id,
            tgt_eos_id=tgt_eos_id,
            tgt_pad_id=tgt_pad_id,
            max_len=max_len,
        )

        for seq in tgt.tolist():
            gold_str = ids_to_string(
                seq,
                tgt_vocab_itos,
                eos_id=tgt_eos_id,
                skip_ids=skip_ids,
            )
            all_gold.append(gold_str)

        all_pred.extend(pred_strs)

    exact = exact_match_accuracy(all_pred, all_gold)

    return (
        total_loss / max(n_batches, 1),
        total_tok_acc / max(n_batches, 1),
        exact,
    )


def save_predictions_tsv(
    save_path: Path,
    pairs: List[tuple[str, str]],
    preds: List[str],
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with save_path.open("w", encoding="utf-8") as f:
        f.write("input\tprediction\tgold\n")

        for (roman, gold), pred in zip(pairs, preds):
            f.write(f"{roman}\t{pred}\t{gold}\n")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--lang", type=str, default="te")
    parser.add_argument("--ckpt_path", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_decode_len", type=int, default=40)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--save_predictions",
        type=str,
        default="predictions_vanilla/preds.tsv",
        help="Path to save all vanilla test predictions.",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data_dir = Path(args.data_dir)

    # Build vocabs the same way training did: from train only
    src_vocab, tgt_vocab = build_vocabs_from_train(data_dir, args.lang)

    # Load checkpoint
    try:
        ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.ckpt_path, map_location=device)

    cfg = ckpt["config"]

    model = VanillaSeq2Seq(
        encoder=Encoder(
            vocab_size=len(src_vocab.itos),
            emb_size=cfg["emb_size"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["enc_layers"],
            cell_type=cfg["cell_type"],
            dropout=cfg["dropout"],
        ),
        decoder=Decoder(
            vocab_size=len(tgt_vocab.itos),
            emb_size=cfg["emb_size"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["dec_layers"],
            cell_type=cfg["cell_type"],
            dropout=cfg["dropout"],
        ),
        pad_id=tgt_vocab.pad_id,
    ).to(device)

    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    # Test loader
    test_loader, test_pairs = build_loader(
        data_dir=data_dir,
        lang=args.lang,
        split="test",
        batch_size=args.batch_size,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_id)

    test_loss, test_tok_acc, test_exact = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        tgt_vocab_itos=tgt_vocab.itos,
        tgt_sos_id=tgt_vocab.sos_id,
        tgt_eos_id=tgt_vocab.eos_id,
        tgt_pad_id=tgt_vocab.pad_id,
        max_len=args.max_decode_len,
    )

    print("\nTEST RESULTS")
    print(f"  test_loss     = {test_loss:.4f}")
    print(f"  test_tok_acc  = {test_tok_acc:.4f}")
    print(f"  test_exact    = {test_exact:.4f}")

    # Save all test predictions
    all_preds = predict_all(
        model=model,
        loader=test_loader,
        device=device,
        tgt_vocab_itos=tgt_vocab.itos,
        tgt_sos_id=tgt_vocab.sos_id,
        tgt_eos_id=tgt_vocab.eos_id,
        tgt_pad_id=tgt_vocab.pad_id,
        max_len=args.max_decode_len,
    )

    save_path = Path(args.save_predictions)
    save_predictions_tsv(save_path, test_pairs, all_preds)
    print(f"\nSaved vanilla predictions to: {save_path}")

    # Sample predictions
    print("\nSAMPLE PREDICTIONS")
    n = min(args.num_samples, len(test_pairs))

    for i in range(n):
        roman, gold = test_pairs[i]
        pred = all_preds[i]

        print(f"IN : {roman}")
        print(f"PRD: {pred}")
        print(f"GLD: {gold}")
        print("-" * 40)


if __name__ == "__main__":
    main()