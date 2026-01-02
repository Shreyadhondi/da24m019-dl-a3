from pathlib import Path

from src.data.load_dakshina import read_lexicon_tsv
from src.data.vocab import build_char_vocab, add_sos_eos


def main():
    train_path = Path("data/te/lexicons/te.translit.sampled.train.tsv")
    split = read_lexicon_tsv(train_path)

    # separate roman and telugu texts
    roman_words = [x for x, _ in split.pairs]
    telugu_words = [y for _, y in split.pairs]

    src_vocab = build_char_vocab(roman_words)
    tgt_vocab = build_char_vocab(telugu_words)

    print("Source vocab size:", len(src_vocab.itos))
    print("Target vocab size:", len(tgt_vocab.itos))

    # take one example
    roman, telugu = split.pairs[0]
    print("\nExample pair:")
    print("roman :", roman)
    print("telugu:", telugu)

    src_ids = add_sos_eos(src_vocab.encode_chars(roman), src_vocab.sos_id, src_vocab.eos_id)
    tgt_ids = add_sos_eos(tgt_vocab.encode_chars(telugu), tgt_vocab.sos_id, tgt_vocab.eos_id)

    print("\nEncoded (with SOS/EOS):")
    print("src_ids:", src_ids)
    print("tgt_ids:", tgt_ids)

    print("\nDecoded back:")
    print("src_decoded:", src_vocab.decode_chars(src_ids))
    print("tgt_decoded:", tgt_vocab.decode_chars(tgt_ids))


if __name__ == "__main__":
    main()
