"""Load a *fast* tokenizer from an uploaded set of files.

Supported inputs
----------------
1. ``tokenizer.json``                      -> tokenizers.Tokenizer (always fast)
2. full HuggingFace fast folder            -> AutoTokenizer(use_fast=True)
   (tokenizer.json + tokenizer_config.json + vocab.json + merges.txt +
    special_tokens_map.json + *.model)
3. BPE ``vocab.json`` + ``merges.txt``     -> manually assembled ByteLevel BPE
4. SentencePiece ``*.model``               -> SentencePiece unigram / BPE

``use_fast=True`` is enforced in every branch; a slow-only tokenizer raises
:class:`NotFastTokenizerError` instead of being silently accepted.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import os

# keep everything local - never reach out to huggingface.co during a request
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class TokenizerLoadError(Exception):
    """Raised when the uploaded files cannot be turned into a usable tokenizer."""


class NotFastTokenizerError(TokenizerLoadError):
    """Raised when only a slow (python) tokenizer could be built."""


# --------------------------------------------------------------------------
# handle
# --------------------------------------------------------------------------

class TokenizerHandle:
    """Uniform wrapper around a ``tokenizers.Tokenizer`` backend."""

    def __init__(
        self,
        backend: Any,
        kind: str,
        files: List[str],
        special_tokens: Optional[Dict[str, Any]] = None,
        warnings: Optional[List[str]] = None,
        hf_tokenizer: Any = None,
        source: str = "",
    ) -> None:
        self.backend = backend
        self.kind = kind
        self.files = files
        self.special_tokens = special_tokens or {}
        self.warnings = warnings or []
        self.hf = hf_tokenizer
        self.source = source
        self._lock = threading.Lock()

    # -- introspection -----------------------------------------------------
    @property
    def vocab_size(self) -> int:
        try:
            return int(self.backend.get_vocab_size())
        except Exception:
            return 0

    @property
    def is_fast(self) -> bool:
        return True  # tokenizers.Tokenizer is by definition the fast backend

    def info(self, session_id: str) -> Dict[str, Any]:
        return {
            "session_id": session_id,
            "kind": self.kind,
            "is_fast": True,
            "vocab_size": self.vocab_size,
            "files": list(self.files),
            "warnings": list(self.warnings),
            "special_tokens": dict(self.special_tokens),
        }

    # -- encoding ----------------------------------------------------------
    def encode_with_offsets(
        self, text: str, add_special_tokens: bool = False
    ) -> Dict[str, Any]:
        """Encode ``text`` and return ids / tokens / character offsets."""
        if not isinstance(text, str):
            text = str(text)
        with self._lock:
            if add_special_tokens and self.hf is not None:
                enc = self.hf(
                    text,
                    add_special_tokens=True,
                    return_offsets_mapping=True,
                )
                ids = list(enc["input_ids"])
                offsets = [tuple(o) for o in enc["offset_mapping"]]
                token_strings = [self._id_to_token(i) for i in ids]
                return {"ids": ids, "offsets": offsets, "tokens": token_strings}

            enc = self.backend.encode(text, add_special_tokens=False)
            return {
                "ids": list(enc.ids),
                "offsets": [tuple(o) for o in enc.offsets],
                "tokens": list(enc.tokens),
            }

    def _id_to_token(self, token_id: int) -> str:
        tok = self.backend.id_to_token(int(token_id))
        return tok if tok is not None else ""

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self._id_to_token(i) for i in ids]

    def decode(self, ids: List[int]) -> str:
        """Decode with special tokens removed (the fidelity check target)."""
        with self._lock:
            try:
                return self.backend.decode([int(i) for i in ids], skip_special_tokens=True)
            except Exception:
                return ""

    def raw_decode(self, ids: List[int]) -> str:
        """Decode keeping special tokens (fallback fidelity comparison)."""
        with self._lock:
            try:
                return self.backend.decode([int(i) for i in ids], skip_special_tokens=False)
            except Exception:
                return ""


# --------------------------------------------------------------------------
# loading strategies
# --------------------------------------------------------------------------

def _read_json(path: Path) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _collect_special_tokens(hf_tokenizer: Any, session_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if hf_tokenizer is not None:
        try:
            for key, value in (hf_tokenizer.special_tokens_map or {}).items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    out[key] = value
                else:
                    out[key] = str(value)
        except Exception:
            pass
    config_path = session_dir / "tokenizer_config.json"
    if config_path.exists():
        cfg = _read_json(config_path) or {}
        if isinstance(cfg, dict):
            for key in (
                "model_max_length",
                "clean_up_tokenization_spaces",
                "tokenizer_class",
                "unk_token",
                "bos_token",
                "eos_token",
                "pad_token",
                "add_bos_token",
                "add_eos_token",
            ):
                if key in cfg:
                    value = cfg[key]
                    if isinstance(value, dict):
                        value = value.get("content", str(value))
                    out.setdefault(key, value)
    if "model_max_length" in out and out["model_max_length"] is None:
        out["model_max_length"] = "unset"
    return out


def _try_huggingface_fast(
    session_dir: Path, warnings: List[str]
) -> Tuple[Any, Any]:
    """Try ``AutoTokenizer.from_pretrained(dir, use_fast=True)``.

    Returns ``(hf_tokenizer, backend)`` or ``(None, None)`` when unavailable.
    """
    try:
        from transformers import AutoTokenizer  # imported lazily
    except Exception as exc:  # pragma: no cover - transformers missing
        warnings.append(f"transformers 不可用：{exc}")
        return None, None

    try:
        try:
            hf_tok = AutoTokenizer.from_pretrained(
                str(session_dir), use_fast=True, local_files_only=True
            )
        except TypeError:
            hf_tok = AutoTokenizer.from_pretrained(str(session_dir), use_fast=True)
    except Exception as exc:
        warnings.append(f"AutoTokenizer 加载失败：{type(exc).__name__}: {exc}")
        return None, None

    if not getattr(hf_tok, "is_fast", False):
        raise NotFastTokenizerError(
            "AutoTokenizer 只解析出 slow (Python) tokenizer，本项目强制要求 fast tokenizer。"
            "请上传 tokenizer.json 或使用支持 fast 实现的模型。"
        )
    backend = getattr(hf_tok, "backend_tokenizer", None)
    if backend is None:
        warnings.append("hf tokenizer 没有 backend_tokenizer，改用 tokenizer.json 直接加载。")
        return None, None
    return hf_tok, backend


def _load_tokenizer_json(session_dir: Path, warnings: List[str]) -> Any:
    from tokenizers import Tokenizer

    path = session_dir / "tokenizer.json"
    return Tokenizer.from_file(str(path))


def _load_manual_bpe(session_dir: Path, warnings: List[str]) -> Any:
    """Assemble a ByteLevel BPE from ``vocab.json`` + ``merges.txt`` (GPT-2 style)."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from tokenizers.processors import ByteLevel as ByteLevelProcessor

    vocab_path = session_dir / "vocab.json"
    merges_path = session_dir / "merges.txt"
    vocab = _read_json(vocab_path)
    if not isinstance(vocab, dict) or not vocab:
        raise TokenizerLoadError("vocab.json 不是合法的 {token: id} 映射。")

    merges: List[Tuple[str, str]] = []
    with open(merges_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if not line or line.startswith("#version"):
                continue
            parts = line.split(" ")
            if len(parts) >= 2:
                merges.append((parts[0], parts[1]))
    if not merges:
        raise TokenizerLoadError("merges.txt 中没有解析到任何 merge 规则。")

    unk = "<unk>" if "<unk>" in vocab else None
    backend = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token=unk))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.post_processor = ByteLevelProcessor(trim_offsets=False)
    warnings.append("手动从 vocab.json + merges.txt 组装了 ByteLevel BPE tokenizer。")
    return backend


def _load_sentencepiece(session_dir: Path, warnings: List[str]) -> Any:
    """Load ``*.model`` (SentencePiece) as a fast tokenizer."""
    from tokenizers import SentencePieceBPETokenizer, SentencePieceUnigramTokenizer

    candidates = sorted(session_dir.glob("*.model"))
    if not candidates:
        raise TokenizerLoadError("没有找到 *.model 文件。")
    model_path = candidates[0]

    model_type = None
    try:
        from sentencepiece import sentencepiece_model_pb2 as model_pb2

        proto = model_pb2.ModelProto()
        with open(model_path, "rb") as fh:
            proto.ParseFromString(fh.read())
        model_type = int(proto.trainer_spec.model_type)
    except Exception as exc:
        warnings.append(f"无法读取 SentencePiece proto（{type(exc).__name__}），将按 unigram 尝试。")

    # 1 = UNIGRAM, 2 = BPE, 3 = WORD, 4 = CHAR
    errors: List[str] = []
    order = []
    if model_type == 2:
        order.append("bpe")
    elif model_type == 1:
        order.append("unigram")
    order.extend(["unigram", "bpe"])

    for kind in dict.fromkeys(order):
        try:
            if kind == "unigram":
                backend = SentencePieceUnigramTokenizer.from_spm(str(model_path))
            else:
                backend = SentencePieceBPETokenizer.from_file(str(model_path))
            warnings.append(f"从 {model_path.name} 加载 SentencePiece {kind} tokenizer。")
            return backend
        except Exception as exc:
            errors.append(f"{kind}: {type(exc).__name__}: {exc}")

    raise TokenizerLoadError(
        "无法从 SentencePiece 模型构建 fast tokenizer（"
        + "；".join(errors)
        + "）。请确认已安装 sentencepiece + protobuf，或改用 tokenizer.json。"
    )


def load_tokenizer_from_dir(session_dir: Path) -> TokenizerHandle:
    """Load the fast tokenizer living in ``session_dir``."""
    session_dir = Path(session_dir)
    if not session_dir.exists() or not session_dir.is_dir():
        raise TokenizerLoadError(f"目录不存在：{session_dir}")

    files = sorted(p.name for p in session_dir.iterdir() if p.is_file())
    if not files:
        raise TokenizerLoadError("上传目录为空，请先上传 tokenizer 文件。")

    warnings: List[str] = []
    names = {name.lower() for name in files}

    # ---- strategy 1: full HF folder (config/tokenizer_config present) -----
    has_config = bool({"tokenizer_config.json", "config.json"} & names)
    if has_config or "tokenizer.json" in names:
        hf_tok, backend = _try_huggingface_fast(session_dir, warnings)
        if backend is not None:
            kind = "hf_folder" if has_config else "tokenizer_json"
            special = _collect_special_tokens(hf_tok, session_dir)
            return TokenizerHandle(
                backend=backend,
                kind=kind,
                files=files,
                special_tokens=special,
                warnings=warnings,
                hf_tokenizer=hf_tok,
                source=str(session_dir),
            )

    # ---- strategy 2: bare tokenizer.json ---------------------------------
    if "tokenizer.json" in names:
        try:
            backend = _load_tokenizer_json(session_dir, warnings)
            return TokenizerHandle(
                backend=backend,
                kind="tokenizer_json",
                files=files,
                warnings=warnings,
                source=str(session_dir),
            )
        except Exception as exc:
            warnings.append(f"tokenizer.json 直接加载失败：{type(exc).__name__}: {exc}")

    # ---- strategy 3: vocab.json + merges.txt (BPE) -----------------------
    if {"vocab.json", "merges.txt"} <= names:
        backend = _load_manual_bpe(session_dir, warnings)
        return TokenizerHandle(
            backend=backend,
            kind="bpe_vocab_merges",
            files=files,
            warnings=warnings,
            source=str(session_dir),
        )

    # ---- strategy 4: SentencePiece *.model -------------------------------
    if any(name.endswith(".model") for name in names):
        backend = _load_sentencepiece(session_dir, warnings)
        return TokenizerHandle(
            backend=backend,
            kind="sentencepiece",
            files=files,
            warnings=warnings,
            source=str(session_dir),
        )

    raise TokenizerLoadError(
        "无法识别的 tokenizer 文件组合："
        + ", ".join(files)
        + "。支持：tokenizer.json；完整 HF fast 目录；vocab.json + merges.txt；*.model。"
    )
