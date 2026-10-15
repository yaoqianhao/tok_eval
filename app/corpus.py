"""Built-in + custom corpus handling.

Layout
------
corpus/builtin/<category>/<subcategory>.txt    built-in seed corpus
corpus/custom/<name>.txt|.jsonl|.csv           user uploaded corpus
corpus/manifest.json                           machine readable inventory

Sample format
-------------
* ``.txt``   one sample per line, UTF-8, blank lines ignored, ``#`` = comment
* ``.jsonl`` one JSON object per line, the sample is read from ``text``
* ``.csv``   a table containing a ``text`` column
"""

from __future__ import annotations

import csv
import io
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
BUILTIN_DIR = ROOT / "corpus" / "builtin"
CUSTOM_DIR = ROOT / "corpus" / "custom"
MANIFEST_PATH = ROOT / "corpus" / "manifest.json"

#: canonical category -> subcategory layout
CATEGORY_SPEC: Dict[str, List[str]] = {
    "human_languages": [
        "english",
        "hindi",
        "chinese",
        "arabic",
        "japanese",
        "german",
        "russian",
        "korean",
    ],
    "programming_languages": ["python", "javascript", "sql", "rust"],
    "scientific_formulas": ["algebra", "calculus", "physics", "statistics"],
    "edge_cases": ["whitespace_control", "long_tokens", "mixed_scripts"],
}

CATEGORY_LABELS: Dict[str, str] = {
    "human_languages": "Human Languages",
    "programming_languages": "Programming Languages",
    "scientific_formulas": "Scientific Formulas",
    "edge_cases": "Edge Cases",
    "custom": "Custom Corpus",
}

_SAFE_NAME = re.compile(r"[^0-9A-Za-z_\-.\u4e00-\u9fff]+")


def safe_slug(value: str, fallback: str = "custom") -> str:
    value = (value or "").strip()
    value = _SAFE_NAME.sub("_", value).strip("._")
    return value or fallback


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def parse_txt(content: str) -> List[str]:
    samples: List[str] = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        samples.append(line)
    return samples


def parse_jsonl(content: str) -> List[str]:
    samples: List[str] = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            value = obj.get("text", obj.get("content", obj.get("sample")))
            if isinstance(value, str) and value.strip():
                samples.append(value)
        elif isinstance(obj, str) and obj.strip():
            samples.append(obj)
    return samples


def parse_csv(content: str) -> List[str]:
    samples: List[str] = []
    reader = csv.DictReader(io.StringIO(content))
    if reader.fieldnames is None:
        return samples
    field = None
    for name in reader.fieldnames:
        if name and name.strip().lower() == "text":
            field = name
            break
    if field is None:
        field = reader.fieldnames[0]
    for row in reader:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            samples.append(value.strip())
    return samples


def parse_bytes(raw: bytes, suffix: str) -> List[str]:
    text = raw.decode("utf-8-sig", errors="replace")
    suffix = (suffix or "").lower().lstrip(".")
    if suffix == "jsonl":
        return parse_jsonl(text)
    if suffix == "csv":
        return parse_csv(text)
    return parse_txt(text)


def parse_corpus_file(path: Path) -> List[str]:
    try:
        raw = path.read_bytes()
    except Exception:
        return []
    return parse_bytes(raw, path.suffix)


# --------------------------------------------------------------------------
# inventory
# --------------------------------------------------------------------------

def _entry(category: str, subcategory: str, path: Path, samples: List[str]) -> Dict[str, Any]:
    return {
        "category": category,
        "category_label": CATEGORY_LABELS.get(category, category),
        "subcategory": subcategory,
        "name": f"{category}/{subcategory}",
        "path": str(path),
        "samples": samples,
        "sample_count": len(samples),
    }


def load_builtin_corpus() -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not BUILTIN_DIR.exists():
        return entries
    for category_dir in sorted(BUILTIN_DIR.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        for corpus_file in sorted(category_dir.glob("*.txt")):
            samples = parse_corpus_file(corpus_file)
            if samples:
                entries.append(_entry(category, corpus_file.stem, corpus_file, samples))
    return entries


def load_custom_corpus() -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not CUSTOM_DIR.exists():
        return entries
    # custom/<category>/<subcategory>.<ext>  or  custom/<file>.<ext>
    for path in sorted(CUSTOM_DIR.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".jsonl", ".csv"}:
            continue
        rel = path.relative_to(CUSTOM_DIR)
        if len(rel.parts) >= 2:
            category = rel.parts[0]
            subcategory = path.stem
        else:
            category = "custom"
            subcategory = path.stem
        samples = parse_corpus_file(path)
        if samples:
            entries.append(_entry(category, subcategory, path, samples))
    return entries


def filter_entries(
    entries: Sequence[Dict[str, Any]],
    categories: Optional[Sequence[str]] = None,
    subcategories: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    out = list(entries)
    if categories:
        wanted = set(categories)
        out = [e for e in out if e["category"] in wanted]
    if subcategories:
        wanted = {s.lower() for s in subcategories}
        out = [e for e in out if e["subcategory"].lower() in wanted]
    return out


def build_manifest() -> Dict[str, Any]:
    """Build (and return) the corpus manifest, also persisted to disk."""
    builtin = load_builtin_corpus()
    custom = load_custom_corpus()
    files: List[Dict[str, Any]] = []
    for entry in builtin + custom:
        files.append(
            {
                "category": entry["category"],
                "subcategory": entry["subcategory"],
                "path": str(Path(entry["path"]).relative_to(ROOT)).replace("\\", "/"),
                "sample_count": entry["sample_count"],
                "source": "custom" if Path(entry["path"]).is_relative_to(CUSTOM_DIR) else "builtin",
            }
        )
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus_root": str(ROOT / "corpus"),
        "builtin_root": str(BUILTIN_DIR),
        "custom_root": str(CUSTOM_DIR),
        "format": {
            "txt": "UTF-8, one sample per line, blank lines ignored, '#' starts a comment",
            "jsonl": 'one JSON object per line, sample read from the "text" field',
            "csv": 'a "text" column, one sample per row',
        },
        "layout": "corpus/builtin/<category>/<subcategory>.txt ; corpus/custom/<category>/<subcategory>.<ext>",
        "categories": CATEGORY_SPEC,
        "category_labels": CATEGORY_LABELS,
        "files": files,
        "builtin_files": len(builtin),
        "custom_files": len(custom),
        "builtin_samples": sum(e["sample_count"] for e in builtin),
        "custom_samples": sum(e["sample_count"] for e in custom),
    }
    try:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass
    return manifest


def save_custom_corpus(
    raw: bytes,
    filename: str,
    category: str = "custom",
    subcategory: str = "custom",
) -> Dict[str, Any]:
    """Persist an uploaded custom corpus file under ``corpus/custom``."""
    suffix = Path(filename).suffix.lower()
    if suffix not in {".txt", ".jsonl", ".csv"}:
        raise ValueError("自定义语料仅支持 .txt / .jsonl / .csv 格式。")

    samples = parse_bytes(raw, suffix)
    if not samples:
        raise ValueError("没有从上传的文件中解析出任何样本，请检查文件格式。")

    category_slug = safe_slug(category, "custom")
    subcategory_slug = safe_slug(subcategory or Path(filename).stem, "custom")
    target_dir = CUSTOM_DIR / category_slug
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{subcategory_slug}{suffix}"
    target.write_bytes(raw)
    build_manifest()
    return {
        "saved_to": str(target),
        "samples": len(samples),
        "category": category_slug,
        "subcategory": subcategory_slug,
    }
