import torch
from pathlib import Path

from src.data.load_dakshina import read_lexicon_tsv
from src.data.vocab import build_char_vocab
from src.data.dataset import DakshinaSeq2SeqDataset, collate_fn
from src.models.vanilla_seq2seq import Encoder, Decoder, VanillaSeq2Seq


def main():
    # --------------------------------------------------
    # Load a small subset of data
    # --------------------------------------------------
    train_path = Path("data/te/lexicons/te.translit.sampled.train.tsv")
    split = read_lexicon_tsv(train_path)

    pairs = split.pairs[:64]  # small subset for testing

    roman_words = [x for x, _ in pairs]
    telugu_words = [y for _, y in pairs]

    src_vocab = build_char_vocab(roman_words)
    tgt_vocab = build_char_vocab(telugu_words)

    dataset = DakshinaSeq2SeqDataset(pairs, src_vocab, tgt_vocab)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, src_vocab.pad_id, tgt_vocab.pad_id),
    )

    src, tgt, src_lens, tgt_lens = next(iter(loader))

    # --------------------------------------------------
    # Build model
    # --------------------------------------------------
    encoder = Encoder(
        vocab_size=len(src_vocab.itos),
        emb_size=32,
        hidden_size=64,
        num_layers=1,
        cell_type="gru",
        dropout=0.0,
    )

    decoder = Decoder(
        vocab_size=len(tgt_vocab.itos),
        emb_size=32,
        hidden_size=64,
        num_layers=1,
        cell_type="gru",
        dropout=0.0,
    )

    model = VanillaSeq2Seq(
        encoder=encoder,
        decoder=decoder,
        pad_id=tgt_vocab.pad_id,
    )

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------
    logits = model(
        src=src,
        src_lens=src_lens,
        tgt=tgt,
        teacher_forcing_ratio=1.0,
    )

    # --------------------------------------------------
    # Print sanity info
    # --------------------------------------------------
    print("Source shape:", tuple(src.shape))
    print("Target shape:", tuple(tgt.shape))
    print("Logits shape:", tuple(logits.shape))
    print("Target vocab size:", len(tgt_vocab.itos))

    # Expected:
    # logits -> [B, T_tgt - 1, V]
    assert logits.shape[0] == src.shape[0]
    assert logits.shape[2] == len(tgt_vocab.itos)

    print("\n Forward pass test PASSED")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
