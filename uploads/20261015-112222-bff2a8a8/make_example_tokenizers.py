"""Build small local fast tokenizers used as upload samples (no network needed).

Usage::

    python examples/make_example_tokenizers.py

Writes:
    examples/example_bpe_small/tokenizer.json     BPE, vocab 300
    examples/example_bpe_large/tokenizer.json     BPE, vocab 800
    examples/example_bpe_vocab_merges/{vocab.json,merges.txt}

The two tokenizers are trained on the very same built-in corpus so that the
Compare tab shows a meaningful difference (bigger vocab -> lower fertility).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402


def load_training_texts() -> list:
    texts = []
    builtin = ROOT / "corpus" / "builtin"
    for path in sorted(builtin.rglob("*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                texts.append(line)
    return texts


def train_bpe(texts, vocab_size: int, out_file: Path) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(texts, trainer=trainer)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_file))
    return tok


def export_vocab_merges(tok: Tokenizer, out_dir: Path) -> None:
    """ByteLevel BPE cannot be exported back to vocab.json/merges.txt by
    ``save`` alone; we write plain JSON / text files for the demo."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab = tok.get_vocab()
    (out_dir / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=0), encoding="utf-8"
    )
    merges = []
    try:
        raw = json.loads(tok.to_str())["model"].get("merges", [])
        for item in raw:
            if isinstance(item, str):
                merges.append(item)
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                merges.append(f"{item[0]} {item[1]}")
    except Exception:
        pass
    (out_dir / "merges.txt").write_text(
        "#version: 0.2\n" + "\n".join(merges) + "\n", encoding="utf-8"
    )


def main() -> None:
    texts = load_training_texts()
    print(f"training samples: {len(texts)}")

    small = train_bpe(texts, 300, ROOT / "examples" / "example_bpe_small" / "tokenizer.json")
    print("vocab small:", small.get_vocab_size())

    large = train_bpe(texts, 800, ROOT / "examples" / "example_bpe_large" / "tokenizer.json")
    print("vocab large:", large.get_vocab_size())

    export_vocab_merges(large, ROOT / "examples" / "example_bpe_vocab_merges")
    print("done.")


if __name__ == "__main__":
    main()
