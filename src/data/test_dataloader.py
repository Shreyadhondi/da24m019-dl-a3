from pathlib import Path

from torch.utils.data import DataLoader

from src.data.load_dakshina import read_lexicon_tsv
from src.data.vocab import build_char_vocab
from src.data.dataset import DakshinaSeq2SeqDataset, collate_fn


def main():
    train_path = Path("data/te/lexicons/te.translit.sampled.train.tsv")
    split = read_lexicon_tsv(train_path)

    roman_words = [x for x, _ in split.pairs]
    telugu_words = [y for _, y in split.pairs]

    src_vocab = build_char_vocab(roman_words)
    tgt_vocab = build_char_vocab(telugu_words)

    ds = DakshinaSeq2SeqDataset(split.pairs[:32], src_vocab, tgt_vocab)

    loader = DataLoader(
        ds,
        batch_size=8,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, src_vocab.pad_id, tgt_vocab.pad_id),
    )

    src, tgt, src_lens, tgt_lens = next(iter(loader))

    print("src shape:", tuple(src.shape))
    print("tgt shape:", tuple(tgt.shape))
    print("src_lens:", src_lens.tolist())
    print("tgt_lens:", tgt_lens.tolist())

    print("\nFirst src row ids:", src[0].tolist())
    print("First tgt row ids:", tgt[0].tolist())


if __name__ == "__main__":
    main()
