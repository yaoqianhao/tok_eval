"""Proof that the loader really enforces ``use_fast=True``.

A slow-only tokenizer (``is_fast == False``) must raise
``NotFastTokenizerError``, which the API surfaces as a clear error message.

Run::

    python examples/verify_fast_enforcement.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import tokenizer_loader as tl  # noqa: E402


class FakeSlowTokenizer:
    """Mimics what ``AutoTokenizer(use_fast=True)`` returns when no fast
    implementation exists: a slow python tokenizer."""

    is_fast = False
    slow = True


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="slow_only_"))
    (tmp / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "FakeSlowTokenizer"}), encoding="utf-8"
    )
    (tmp / "vocab.txt").write_text("[PAD]\n[UNK]\nhello\nworld\n", encoding="utf-8")

    with mock.patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=FakeSlowTokenizer(),
    ):
        try:
            tl.load_tokenizer_from_dir(tmp)
        except tl.NotFastTokenizerError as exc:
            print("PASS: slow-only tokenizer rejected ->", exc)
            return 0
        except Exception as exc:  # pragma: no cover
            print("FAIL: unexpected exception ->", type(exc).__name__, exc)
            return 1
        print("FAIL: no error raised for a slow-only tokenizer")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
