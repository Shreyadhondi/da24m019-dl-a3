from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from src.data.load_dakshina import read_lexicon_tsv
from src.data.vocab import build_char_vocab
from src.data.dataset import DakshinaSeq2SeqDataset, collate_fn
from src.models.attention import Encoder, AttentionDecoder, AttnSeq2Seq


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


def build_test_loader_and_vocabs(
    data_dir: Path, lang: str, batch_size: int
) -> Tuple[DataLoader, Dict, Dict, List[Tuple[str, str]]]:
    train_path = data_dir / lang / "lexicons" / f"{lang}.translit.sampled.train.tsv"
    test_path = data_dir / lang / "lexicons" / f"{lang}.translit.sampled.test.tsv"

    train_split = read_lexicon_tsv(train_path)
    test_split = read_lexicon_tsv(test_path)

    train_romans = [x for x, _ in train_split.pairs]
    train_natives = [y for _, y in train_split.pairs]

    src_vocab = build_char_vocab(train_romans)
    tgt_vocab = build_char_vocab(train_natives)

    test_ds = DakshinaSeq2SeqDataset(test_split.pairs, src_vocab, tgt_vocab)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, src_vocab.pad_id, tgt_vocab.pad_id),
        num_workers=0,
    )

    return test_loader, src_vocab, tgt_vocab, test_split.pairs


def safe_torch_load(path: str, device: torch.device) -> Dict:
    # Avoid the FutureWarning when possible (new torch versions support weights_only)
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def greedy_decode_with_attn(
    model: AttnSeq2Seq,
    src: torch.Tensor,
    src_lens: torch.Tensor,
    tgt_sos_id: int,
    tgt_eos_id: int,
    max_len: int,
) -> Tuple[List[List[int]], List[List[List[float]]]]:
    """
    Returns:
      pred_ids: list of length B, each a list of predicted token ids (until EOS)
      attn: list of length B, each a list over time, each is list over src positions
    """
    model.eval()
    enc_out, enc_state = model.encoder(src, src_lens)
    # single layer expected
    if isinstance(enc_state, tuple):
        enc_state = (enc_state[0][:1], enc_state[1][:1])
    else:
        enc_state = enc_state[:1]

    B = src.size(0)
    inp = torch.full((B,), tgt_sos_id, dtype=torch.long, device=src.device)
    state = enc_state

    preds: List[List[int]] = [[] for _ in range(B)]
    attn_all: List[List[List[float]]] = [[] for _ in range(B)]
    finished = torch.zeros((B,), dtype=torch.bool, device=src.device)

    for _t in range(max_len):
        logits, state, attn = model.decoder.forward_step(inp, state, enc_out, src_lens)  # attn [B,Tsrc]
        nxt = logits.argmax(dim=1)
        inp = nxt

        attn_cpu = attn.detach().cpu().tolist()
        for i in range(B):
            if not finished[i]:
                preds[i].append(int(nxt[i].item()))
                attn_all[i].append(attn_cpu[i])
                if int(nxt[i].item()) == tgt_eos_id:
                    finished[i] = True

        if bool(finished.all()):
            break

    return preds, attn_all


def exact_match_accuracy(preds: List[str], gold: List[str]) -> float:
    if len(gold) == 0:
        return 0.0
    return sum(p == g for p, g in zip(preds, gold)) / len(gold)


def token_accuracy_from_strings(pred_ids: List[List[int]], gold_ids: List[List[int]], pad_id: int) -> float:
    # approximate token acc on decoded sequences vs gold (up to min length)
    total = 0
    correct = 0
    for p, g in zip(pred_ids, gold_ids):
        # gold ids include <sos> at start, so compare with g[1:]
        g2 = []
        for x in g[1:]:
            if x == pad_id:
                break
            g2.append(x)

        T = min(len(p), len(g2))
        for t in range(T):
            total += 1
            if p[t] == g2[t]:
                correct += 1
    return correct / max(total, 1)


def save_heatmap(
    attn: List[List[float]],   # [Tpred, Tsrc]
    src_str: str,
    pred_str: str,
    out_path: Path,
) -> None:
    # make figure (no explicit colors specified)
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)

    mat = torch.tensor(attn)
    ax.imshow(mat.numpy(), aspect="auto")

    ax.set_title("Attention heatmap")
    ax.set_xlabel("Source positions")
    ax.set_ylabel("Decoder steps")

    # annotate with strings (kept short to avoid clutter)
    ax.text(0.0, -0.15, f"SRC: {src_str}", transform=ax.transAxes)
    ax.text(0.0, -0.28, f"PRD: {pred_str}", transform=ax.transAxes)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--lang", type=str, default="te")
    parser.add_argument("--ckpt_path", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_decode_len", type=int, default=40)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--save_predictions", type=str, default="predictions_attention/preds.tsv")
    parser.add_argument("--save_heatmaps_dir", type=str, default="attention_heatmaps")

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_loader, src_vocab, tgt_vocab, test_pairs = build_test_loader_and_vocabs(
        data_dir=Path(args.data_dir),
        lang=args.lang,
        batch_size=args.batch_size,
    )

    ckpt = safe_torch_load(args.ckpt_path, device=device)
    cfg = ckpt.get("config", {})

    emb_size = int(cfg.get("emb_size", 64))
    hidden_size = int(cfg.get("hidden_size", 128))
    cell_type = str(cfg.get("cell_type", "gru"))
    dropout = float(cfg.get("dropout", 0.2))

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
    model = AttnSeq2Seq(encoder=encoder, decoder=decoder, pad_id=tgt_vocab.pad_id).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_id)

    total_loss = 0.0
    n_batches = 0

    all_pred_str: List[str] = []
    all_gold_str: List[str] = []
    all_inp_str: List[str] = []

    # for token acc from ids
    all_pred_ids: List[List[int]] = []
    all_gold_ids: List[List[int]] = []

    skip_ids = {tgt_vocab.sos_id, tgt_vocab.pad_id}

    heatmaps_saved = 0
    heatmap_dir = Path(args.save_heatmaps_dir)
    pred_tsv_path = Path(args.save_predictions)
    pred_tsv_path.parent.mkdir(parents=True, exist_ok=True)

    for src, tgt, src_lens, tgt_lens in test_loader:
        src = src.to(device)
        tgt = tgt.to(device)
        src_lens = src_lens.to(device)

        # loss with teacher forcing
        logits = model(src, src_lens, tgt, teacher_forcing_ratio=1.0)
        targets = tgt[:, 1:]
        B, Tm1, V = logits.shape
        loss = criterion(logits.reshape(B * Tm1, V), targets.reshape(B * Tm1))
        total_loss += float(loss.item())
        n_batches += 1

        # decode with attention
        pred_ids_batch, attn_batch = greedy_decode_with_attn(
            model=model,
            src=src,
            src_lens=src_lens,
            tgt_sos_id=tgt_vocab.sos_id,
            tgt_eos_id=tgt_vocab.eos_id,
            max_len=args.max_decode_len,
        )

        # build strings
        src_ids = src.detach().cpu().tolist()
        tgt_ids = tgt.detach().cpu().tolist()

        for i in range(len(pred_ids_batch)):
            # input roman string from src ids
            inp_str = ids_to_string(
                src_ids[i],
                itos=src_vocab.itos,
                eos_id=src_vocab.eos_id,
                skip_ids={src_vocab.sos_id, src_vocab.pad_id},
            )

            prd_str = ids_to_string(
                pred_ids_batch[i],
                itos=tgt_vocab.itos,
                eos_id=tgt_vocab.eos_id,
                skip_ids=skip_ids,
            )

            gld_str = ids_to_string(
                tgt_ids[i],
                itos=tgt_vocab.itos,
                eos_id=tgt_vocab.eos_id,
                skip_ids=skip_ids,
            )

            all_inp_str.append(inp_str)
            all_pred_str.append(prd_str)
            all_gold_str.append(gld_str)

            all_pred_ids.append(pred_ids_batch[i])
            all_gold_ids.append(tgt_ids[i])

            # save up to 10 heatmaps
            if heatmaps_saved < 10:
                out_path = heatmap_dir / f"heatmap_{heatmaps_saved+1:02d}.png"
                save_heatmap(attn_batch[i], inp_str, prd_str, out_path)
                heatmaps_saved += 1

    test_loss = total_loss / max(n_batches, 1)
    test_exact = exact_match_accuracy(all_pred_str, all_gold_str)
    test_tok_acc = token_accuracy_from_strings(all_pred_ids, all_gold_ids, pad_id=tgt_vocab.pad_id)

    print("\nTEST RESULTS")
    print(f"  test_loss     = {test_loss:.4f}")
    print(f"  test_tok_acc  = {test_tok_acc:.4f}")
    print(f"  test_exact    = {test_exact:.4f}")

    # Save predictions TSV
    with pred_tsv_path.open("w", encoding="utf-8") as f:
        f.write("input\tpred\tgold\n")
        for inp, prd, gld in zip(all_inp_str, all_pred_str, all_gold_str):
            f.write(f"{inp}\t{prd}\t{gld}\n")

    # Show sample predictions
    print("\nSAMPLE PREDICTIONS")
    for i in range(min(args.num_samples, len(all_inp_str))):
        print(f"IN : {all_inp_str[i]}")
        print(f"PRD: {all_pred_str[i]}")
        print(f"GLD: {all_gold_str[i]}")
        print("-" * 40)

    print("\nSaved predictions to:", pred_tsv_path)
    print("Saved attention heatmaps to:", heatmap_dir.resolve())


if __name__ == "__main__":
    main()
