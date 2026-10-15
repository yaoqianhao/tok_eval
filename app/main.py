"""tok_eval - FastAPI backend.

Run with::

    python -m uvicorn app.main:app --port 8000

Endpoints
---------
GET  /                      frontend (web/index.html)
GET  /api/health            liveness + inventory
GET  /api/sessions          uploaded tokenizers of this server run
GET  /api/corpus            corpus manifest
POST /api/corpus/custom     upload a custom corpus (.txt / .jsonl / .csv)
POST /api/upload            upload tokenizer files -> session_id
POST /api/playground        tokenize one text + fidelity round-trip
POST /api/evaluate          full evaluation of one tokenizer
POST /api/compare           side by side comparison of two tokenizers
GET  /api/export/{run_id}   export a stored run as csv / json
GET  /api/result/{run_id}   re-read a stored run
"""

from __future__ import annotations

import csv
import io
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__
from . import corpus as corpus_mod
from . import metrics
from .schemas import (
    CompareRequest,
    CompareResponse,
    CustomCorpusResponse,
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
    PlaygroundRequest,
    PlaygroundResponse,
    TokenPiece,
    TokenizerInfo,
    UploadResponse,
)
from .tokenizer_loader import (
    NotFastTokenizerError,
    TokenizerHandle,
    TokenizerLoadError,
    load_tokenizer_from_dir,
)

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
UPLOAD_DIR = ROOT / "uploads"
RESULTS_DIR = ROOT / "results"

for _d in (UPLOAD_DIR, RESULTS_DIR, corpus_mod.CUSTOM_DIR):
    _d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="tok_eval", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# in-process state
# --------------------------------------------------------------------------
_HANDLES: Dict[str, TokenizerHandle] = {}
_LABELS: Dict[str, str] = {}
_RUNS: Dict[str, Dict[str, Any]] = {}

MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB per file


def _new_session_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]


def _new_run_id(prefix: str) -> str:
    return f"{prefix}-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _handle(session_id: str) -> TokenizerHandle:
    handle = _HANDLES.get(session_id)
    if handle is None:
        raise HTTPException(
            status_code=404,
            detail=f"session {session_id} 不存在或尚未加载 tokenizer，请先上传。",
        )
    return handle


def _label(session_id: str) -> str:
    return _LABELS.get(session_id) or session_id


def _info(session_id: str) -> TokenizerInfo:
    handle = _handle(session_id)
    return TokenizerInfo(**handle.info(session_id))


# --------------------------------------------------------------------------
# frontend
# --------------------------------------------------------------------------
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/", include_in_schema=False)
def index() -> Response:
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        return JSONResponse({"error": "web/index.html 不存在"}, status_code=500)
    return FileResponse(str(index_path))


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    ico_path = WEB_DIR / "favicon.ico"
    if ico_path.exists():
        return FileResponse(str(ico_path), media_type="image/x-icon")
    return Response(status_code=204)


# --------------------------------------------------------------------------
# meta
# --------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    manifest = corpus_mod.build_manifest()
    return HealthResponse(
        status="ok",
        version=__version__,
        builtin_corpus_files=int(manifest.get("builtin_files", 0)),
        sessions=len(_HANDLES),
    )


@app.get("/api/corpus")
def api_corpus() -> Dict[str, Any]:
    manifest = corpus_mod.build_manifest()
    entries = corpus_mod.load_builtin_corpus() + corpus_mod.load_custom_corpus()
    detail = [
        {
            "name": e["name"],
            "category": e["category"],
            "category_label": e["category_label"],
            "subcategory": e["subcategory"],
            "sample_count": e["sample_count"],
            "path": e["path"],
            "preview": e["samples"][:2],
        }
        for e in entries
    ]
    manifest["entries"] = detail
    return manifest


@app.post("/api/corpus/custom", response_model=CustomCorpusResponse)
async def api_custom_corpus(
    file: UploadFile = File(...),
    category: str = Form("custom"),
    subcategory: str = Form(""),
) -> CustomCorpusResponse:
    raw = await file.read()
    try:
        result = corpus_mod.save_custom_corpus(
            raw,
            file.filename or "custom.txt",
            category=category or "custom",
            subcategory=subcategory or Path(file.filename or "custom").stem,
        )
    except Exception as exc:
        return CustomCorpusResponse(ok=False, error=str(exc))
    return CustomCorpusResponse(ok=True, saved_to=result["saved_to"], samples=result["samples"])


@app.get("/api/sessions")
def api_sessions() -> Dict[str, Any]:
    sessions = []
    for sid, handle in _HANDLES.items():
        sessions.append(
            {
                "session_id": sid,
                "label": _label(sid),
                "kind": handle.kind,
                "vocab_size": handle.vocab_size,
                "is_fast": True,
            }
        )
    sessions.sort(key=lambda s: s["session_id"], reverse=True)
    return {"sessions": sessions}


# --------------------------------------------------------------------------
# upload
# --------------------------------------------------------------------------
@app.post("/api/upload", response_model=UploadResponse)
async def api_upload(
    files: List[UploadFile] = File(...),
    label: str = Form(""),
) -> UploadResponse:
    if not files:
        return UploadResponse(ok=False, error="没有收到任何文件。")

    session_id = _new_session_id()
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    saved: List[str] = []
    for upload in files:
        raw_name = upload.filename or ""
        name = Path(raw_name.replace("\\", "/")).name  # strip any directory part
        if not name or name in {".", ".."}:
            continue
        data = await upload.read()
        if len(data) > MAX_UPLOAD_BYTES:
            return UploadResponse(ok=False, error=f"{name} 超过 200MB 上限。")
        (session_dir / name).write_bytes(data)
        saved.append(name)

    if not saved:
        return UploadResponse(ok=False, error="没有有效的文件名。")

    try:
        handle = load_tokenizer_from_dir(session_dir)
    except NotFastTokenizerError as exc:
        return UploadResponse(ok=False, error=f"非 fast tokenizer：{exc}")
    except TokenizerLoadError as exc:
        return UploadResponse(ok=False, error=str(exc))
    except Exception as exc:  # pragma: no cover
        return UploadResponse(ok=False, error=f"{type(exc).__name__}: {exc}")

    _HANDLES[session_id] = handle
    _LABELS[session_id] = (label or "").strip() or f"tokenizer-{session_id[-4:]}"
    return UploadResponse(ok=True, tokenizer=TokenizerInfo(**handle.info(session_id)))


# --------------------------------------------------------------------------
# playground
# --------------------------------------------------------------------------
@app.post("/api/playground", response_model=PlaygroundResponse)
def api_playground(req: PlaygroundRequest) -> PlaygroundResponse:
    handle = _handle(req.session_id)
    text = req.text or ""
    if not text:
        return PlaygroundResponse(ok=True, decoded="", error=None)

    enc = handle.encode_with_offsets(text, add_special_tokens=req.include_special_tokens)
    ids: List[int] = enc["ids"]
    offsets: List[Tuple[int, int]] = enc["offsets"]
    tokens: List[str] = enc["tokens"]

    spans = metrics.word_spans(text)
    pieces: List[TokenPiece] = []
    for idx, (token_id, (start, end), token_str) in enumerate(zip(ids, offsets, tokens)):
        is_cont = False
        if end > start:
            for ws, we in spans:
                if ws <= start < we:
                    if start != ws or metrics.is_marked_continuation(token_str):
                        is_cont = True
                    break
        else:
            is_cont = metrics.is_marked_continuation(token_str)
        pieces.append(
            TokenPiece(
                index=idx,
                token=token_str,
                surface=text[start:end] if end > start else "",
                id=int(token_id),
                start=int(start),
                end=int(end),
                is_continuation=is_cont,
            )
        )

    decoded = handle.decode(ids)
    fidelity = metrics._fidelity_ok(text, decoded, handle, ids)
    char_count = len(text)
    byte_count = len(text.encode("utf-8"))
    word_count = len(spans)
    count = len(ids)

    return PlaygroundResponse(
        ok=True,
        tokens=pieces,
        ids=ids,
        count=count,
        char_count=char_count,
        byte_count=byte_count,
        word_count=word_count,
        fertility=count / word_count if word_count else 0.0,
        compression_ratio=count / char_count if char_count else 0.0,
        byte_compression=count / byte_count if byte_count else 0.0,
        decoded=decoded,
        fidelity_pass=bool(fidelity),
    )


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def _select_entries(
    categories: Optional[Sequence[str]] = None,
    subcategories: Optional[Sequence[str]] = None,
    include_custom: bool = True,
    custom_only: bool = False,
) -> List[Dict[str, Any]]:
    if custom_only:
        entries = corpus_mod.load_custom_corpus()
    else:
        entries = corpus_mod.load_builtin_corpus()
        if include_custom:
            entries += corpus_mod.load_custom_corpus()
    return corpus_mod.filter_entries(entries, categories, subcategories)


def _clean_metrics(m: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in m.items() if k != "failed_samples"}
    return out


def _evaluate_all(
    handle: TokenizerHandle,
    entries: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    all_samples: List[str] = []
    for entry in entries:
        all_samples.extend(entry["samples"])
        m = _clean_metrics(metrics.evaluate_samples(handle, entry["samples"]))
        entry_path = Path(entry["path"])
        try:
            is_custom = entry_path.is_relative_to(corpus_mod.CUSTOM_DIR)
        except Exception:
            is_custom = False
        row = {
            "category": entry["category"],
            "category_label": entry["category_label"],
            "subcategory": entry["subcategory"],
            "name": entry["name"],
            "source": "custom" if is_custom else "builtin",
            "sample_count": entry["sample_count"],
            **m,
        }
        rows.append(row)

    summary = _clean_metrics(metrics.evaluate_samples(handle, all_samples))

    # cross language fairness: token cost = tokens / raw UTF-8 byte, languages evenly weighted
    lang_cost: Dict[str, float] = {}
    for row in rows:
        if row["category"] == "human_languages" and row["byte_compression"] > 0:
            lang_cost[row["subcategory"]] = float(row["byte_compression"])
    summary["per_language_token_cost"] = lang_cost
    summary["fairness_gini"] = metrics.fairness_gini(lang_cost)
    summary["composite_score"] = metrics.composite_score(summary)

    charts = {
        "fertility_by_subcategory": {
            "labels": [f'{r["subcategory"]}' for r in rows],
            "full_labels": [r["name"] for r in rows],
            "values": [round(float(r["fertility"]), 4) for r in rows],
            "categories": [r["category_label"] for r in rows],
        },
        "language_fertility": {
            "labels": [r["subcategory"] for r in rows if r["category"] == "human_languages"],
            "values": [
                round(float(r["fertility"]), 4) for r in rows if r["category"] == "human_languages"
            ],
        },
        "scatter": {
            "points": [
                {
                    "x": round(float(r["compression_ratio"]), 4),
                    "y": round(float(r["fertility"]), 4),
                    "label": r["name"],
                }
                for r in rows
            ]
        },
        "cards": {
            "fertility": round(float(summary.get("fertility", 0.0)), 3),
            "compression_ratio": round(float(summary.get("compression_ratio", 0.0)), 4),
            "byte_compression": round(float(summary.get("byte_compression", 0.0)), 4),
            "fidelity_pass_rate": round(float(summary.get("fidelity_pass_rate", 0.0)), 4),
            "shannon_entropy": round(float(summary.get("shannon_entropy", 0.0)), 3),
            "fairness_gini": round(float(summary.get("fairness_gini", 0.0)), 4),
            "vocabulary_utilization": round(float(summary.get("vocabulary_utilization", 0.0)), 4),
            "composite_score": summary.get("composite_score", 0.0),
        },
    }
    return summary, rows, charts


@app.post("/api/evaluate", response_model=EvaluateResponse)
def api_evaluate(req: EvaluateRequest) -> EvaluateResponse:
    handle = _handle(req.session_id)
    entries = _select_entries(
        req.categories, req.subcategories, req.include_custom, req.custom_only
    )
    if not entries:
        return EvaluateResponse(ok=False, error="没有匹配到任何语料条目，请检查筛选条件。")

    summary, rows, charts = _evaluate_all(handle, entries)
    run_id = req.run_id or _new_run_id("eval")
    payload = {
        "run_id": run_id,
        "type": "evaluate",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "session_id": req.session_id,
        "label": _label(req.session_id),
        "tokenizer": _info(req.session_id).model_dump(),
        "summary": summary,
        "rows": rows,
    }
    _store_run(run_id, payload)
    return EvaluateResponse(
        ok=True,
        run_id=run_id,
        tokenizer=_info(req.session_id),
        summary=summary,
        rows=rows,
        charts=charts,
    )


# --------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------
@app.post("/api/compare", response_model=CompareResponse)
def api_compare(req: CompareRequest) -> CompareResponse:
    if req.session_a == req.session_b:
        return CompareResponse(ok=False, error="请选择两个不同的 tokenizer。")
    handle_a = _handle(req.session_a)
    handle_b = _handle(req.session_b)

    entries = _select_entries(req.categories, None, req.include_custom, False)
    if not entries:
        return CompareResponse(ok=False, error="没有匹配到任何语料条目。")

    summary_a, rows_a, _charts_a = _evaluate_all(handle_a, entries)
    summary_b, rows_b, _charts_b = _evaluate_all(handle_b, entries)

    label_a = _label(req.session_a)
    label_b = _label(req.session_b)

    by_name_a = {r["name"]: r for r in rows_a}
    by_name_b = {r["name"]: r for r in rows_b}
    names = [e["name"] for e in entries]

    grouped = {
        "labels": names,
        "subcategories": [e["subcategory"] for e in entries],
        "fertility_a": [round(float(by_name_a[n]["fertility"]), 4) for n in names],
        "fertility_b": [round(float(by_name_b[n]["fertility"]), 4) for n in names],
        "byte_compression_a": [round(float(by_name_a[n]["byte_compression"]), 4) for n in names],
        "byte_compression_b": [round(float(by_name_b[n]["byte_compression"]), 4) for n in names],
        "language_labels": [e["subcategory"] for e in entries if e["category"] == "human_languages"],
        "language_fertility_a": [
            round(float(by_name_a[e["name"]]["fertility"]), 4)
            for e in entries
            if e["category"] == "human_languages"
        ],
        "language_fertility_b": [
            round(float(by_name_b[e["name"]]["fertility"]), 4)
            for e in entries
            if e["category"] == "human_languages"
        ],
    }

    def _lb(label: str, sid: str, summary: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        worst = str(max(rows, key=lambda r: float(r["fertility"]))["name"]) if rows else ""
        return {
            "label": label,
            "session_id": sid,
            "vocab_size": summary.get("vocab_size", 0),
            "samples_evaluated": summary.get("samples_evaluated", 0),
            "composite_score": summary.get("composite_score", 0.0),
            "avg_tokens": round(float(summary.get("avg_tokens", 0.0)), 3),
            "fertility": round(float(summary.get("fertility", 0.0)), 4),
            "fertility_grade": summary.get("fertility_grade", "n/a"),
            "compression_ratio": round(float(summary.get("compression_ratio", 0.0)), 4),
            "byte_compression": round(float(summary.get("byte_compression", 0.0)), 4),
            "fidelity_pass_rate": round(float(summary.get("fidelity_pass_rate", 0.0)), 4),
            "fidelity_fails": summary.get("fidelity_fails", 0),
            "avg_token_bytes": round(float(summary.get("avg_token_bytes", 0.0)), 3),
            "type_token_ratio": round(float(summary.get("type_token_ratio", 0.0)), 4),
            "vocabulary_utilization": round(float(summary.get("vocabulary_utilization", 0.0)), 4),
            "unique_tokens": summary.get("unique_tokens", 0),
            "subword_fertility": round(float(summary.get("subword_fertility", 0.0)), 4),
            "continued_word_rate": round(float(summary.get("continued_word_rate", 0.0)), 4),
            "shannon_entropy": round(float(summary.get("shannon_entropy", 0.0)), 4),
            "renyi_entropy_alpha_0_5": round(float(summary.get("renyi_entropy_alpha_0_5", 0.0)), 4),
            "renyi_entropy_alpha_1": round(float(summary.get("renyi_entropy_alpha_1", 0.0)), 4),
            "renyi_entropy_alpha_2": round(float(summary.get("renyi_entropy_alpha_2", 0.0)), 4),
            "fairness_gini": round(float(summary.get("fairness_gini", 0.0)), 4),
            "worst_subcategory": worst,
        }

    leaderboard = [_lb(label_a, req.session_a, summary_a, rows_a), _lb(label_b, req.session_b, summary_b, rows_b)]
    leaderboard.sort(key=lambda r: float(r["composite_score"]), reverse=True)

    charts = {
        "grouped_bar": {
            "labels": grouped["subcategories"],
            "full_labels": names,
            "series_a": grouped["fertility_a"],
            "series_b": grouped["fertility_b"],
            "label_a": label_a,
            "label_b": label_b,
        },
        "language_bar": {
            "labels": grouped["language_labels"],
            "series_a": grouped["language_fertility_a"],
            "series_b": grouped["language_fertility_b"],
            "label_a": label_a,
            "label_b": label_b,
        },
        "scatter": {
            "points_a": [
                {
                    "x": round(float(r["compression_ratio"]), 4),
                    "y": round(float(r["fertility"]), 4),
                    "label": r["name"],
                }
                for r in rows_a
            ],
            "points_b": [
                {
                    "x": round(float(r["compression_ratio"]), 4),
                    "y": round(float(r["fertility"]), 4),
                    "label": r["name"],
                }
                for r in rows_b
            ],
            "label_a": label_a,
            "label_b": label_b,
        },
        "cards": {
            "label_a": label_a,
            "label_b": label_b,
            "score_a": summary_a.get("composite_score", 0.0),
            "score_b": summary_b.get("composite_score", 0.0),
            "fertility_a": round(float(summary_a.get("fertility", 0.0)), 3),
            "fertility_b": round(float(summary_b.get("fertility", 0.0)), 3),
            "byte_compression_a": round(float(summary_a.get("byte_compression", 0.0)), 4),
            "byte_compression_b": round(float(summary_b.get("byte_compression", 0.0)), 4),
            "fidelity_a": round(float(summary_a.get("fidelity_pass_rate", 0.0)), 4),
            "fidelity_b": round(float(summary_b.get("fidelity_pass_rate", 0.0)), 4),
        },
    }

    run_id = req.run_id or _new_run_id("cmp")
    payload = {
        "run_id": run_id,
        "type": "compare",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "session_a": req.session_a,
        "session_b": req.session_b,
        "label_a": label_a,
        "label_b": label_b,
        "leaderboard": leaderboard,
        "grouped": grouped,
        "summary_a": summary_a,
        "summary_b": summary_b,
        "rows_a": rows_a,
        "rows_b": rows_b,
    }
    _store_run(run_id, payload)
    return CompareResponse(
        ok=True,
        run_id=run_id,
        label_a=label_a,
        label_b=label_b,
        leaderboard=leaderboard,
        grouped=grouped,
        charts=charts,
    )


# --------------------------------------------------------------------------
# results / export
# --------------------------------------------------------------------------
def _store_run(run_id: str, payload: Dict[str, Any]) -> None:
    _RUNS[run_id] = payload
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / f"{run_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


def _load_run(run_id: str) -> Optional[Dict[str, Any]]:
    if run_id in _RUNS:
        return _RUNS[run_id]
    path = RESULTS_DIR / f"{run_id}.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            _RUNS[run_id] = payload
            return payload
        except Exception:
            return None
    return None


@app.get("/api/result/{run_id}")
def api_result(run_id: str) -> Dict[str, Any]:
    payload = _load_run(run_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} 不存在。")
    return payload


EVAL_COLUMNS = [
    "category",
    "subcategory",
    "name",
    "sample_count",
    "samples_evaluated",
    "avg_tokens",
    "fertility",
    "fertility_grade",
    "compression_ratio",
    "byte_compression",
    "fidelity_pass_rate",
    "fidelity_fails",
    "avg_token_bytes",
    "avg_token_chars",
    "type_token_ratio",
    "vocabulary_utilization",
    "unique_tokens",
    "vocab_size",
    "subword_fertility",
    "continued_word_rate",
    "shannon_entropy",
    "renyi_entropy_alpha_0_5",
    "renyi_entropy_alpha_1",
    "renyi_entropy_alpha_2",
]

COMPARE_COLUMNS = [
    "tokenizer",
    "session_id",
    "subcategory",
    "name",
    "fertility",
    "compression_ratio",
    "byte_compression",
    "fidelity_pass_rate",
    "subword_fertility",
    "continued_word_rate",
]


def _rows_to_csv(columns: Sequence[str], rows: Sequence[Dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(columns), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: row.get(c, "") for c in columns})
    return buf.getvalue()


@app.get("/api/export/{run_id}")
def api_export(run_id: str, format: str = Query("csv", alias="format")) -> Response:
    payload = _load_run(run_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} 不存在。")

    fmt = format.lower()
    if fmt == "json":
        body = json.dumps(payload, ensure_ascii=False, indent=2)
        return Response(
            content=body,
            media_type="application/json; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{run_id}.json"',
            },
        )

    if payload.get("type") == "compare":
        rows: List[Dict[str, Any]] = []
        for key, label in (("rows_a", payload.get("label_a")), ("rows_b", payload.get("label_b"))):
            sid = payload.get("session_a") if key == "rows_a" else payload.get("session_b")
            for row in payload.get(key, []):
                rows.append({**row, "tokenizer": label, "session_id": sid})
        columns: Sequence[str] = COMPARE_COLUMNS
    else:
        rows = payload.get("rows", [])
        columns = EVAL_COLUMNS

    body = _rows_to_csv(columns, rows)
    return Response(
        content=body,
        media_type="text/csv; charset=utf-8-sig",
        headers={"Content-Disposition": f'attachment; filename="{run_id}.csv"'},
    )


@app.get("/api/runs")
def api_runs(limit: int = 20) -> Dict[str, Any]:
    runs = []
    for run_id, payload in list(_RUNS.items())[-limit:]:
        runs.append(
            {
                "run_id": run_id,
                "type": payload.get("type"),
                "created_at": payload.get("created_at"),
                "label": payload.get("label") or payload.get("label_a", ""),
            }
        )
    return {"runs": runs}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
