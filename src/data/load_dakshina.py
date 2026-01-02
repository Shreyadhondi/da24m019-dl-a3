from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union


@dataclass
class DakshinaSplit:
    """Holds (roman, native) word pairs for one split (train/dev/test)."""
    pairs: List[Tuple[str, str]]


def read_lexicon_tsv(path: Union[str, Path]) -> DakshinaSplit:
    """
    Reads a Dakshina lexicon TSV file.

    Expected format per line (usually 3 columns):
        native_script<TAB>romanized<TAB>count

    Returns:
        DakshinaSplit with pairs as:
            (romanized, native_script)
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    pairs: List[Tuple[str, str]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                raise ValueError(
                    f"Bad line (expected at least 2 tab-separated columns) "
                    f"at {path}:{line_no} -> {line!r}"
                )

            native = parts[0].strip()
            roman = parts[1].strip()

            # Keep only valid non-empty pairs
            if roman and native:
                pairs.append((roman, native))

    return DakshinaSplit(pairs=pairs)


if __name__ == "__main__":
    # Update this path to match your project structure:
    # If you moved dataset to: data/te/lexicons/...
    train_path = Path("data/te/lexicons/te.translit.sampled.train.tsv")

    print("Looking for:", train_path.resolve())
    print("Exists?:", train_path.exists())

    data = read_lexicon_tsv(train_path)
    print("Total pairs:", len(data.pairs))
    print("First 5 pairs:")
    for pair in data.pairs[:5]:
        print(pair)
