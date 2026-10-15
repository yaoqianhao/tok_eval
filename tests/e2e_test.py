"""End-to-end HTTP smoke test for Tokenizer Bench (run while the server is up).

    python tests/e2e_test.py

It uploads the example tokenizers from ``examples/``, exercises every public
endpoint (playground / evaluate / compare / export / custom corpus / error
path) and dumps the raw JSON answers into ``tests/artifacts/``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_DIR = HERE / "artifacts"
BOUNDARY = "----TokenizerBenchBoundary"


def post(path, payload):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read().decode("utf-8"))


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=300) as r:
        return r.read().decode("utf-8")


def upload_multipart(path, files=(), fields=None, field_name="files"):
    body = b""
    for f in files:
        f = Path(f)
        body += f"--{BOUNDARY}\r\n".encode()
        body += (
            f'Content-Disposition: form-data; name="{field_name}"; filename="{f.name}"\r\n'
        ).encode()
        body += b"Content-Type: application/octet-stream\r\n\r\n"
        body += f.read_bytes() + b"\r\n"
    for key, value in (fields or {}).items():
        body += f"--{BOUNDARY}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode()
        body += str(value).encode("utf-8") + b"\r\n"
    body += f"--{BOUNDARY}--\r\n".encode()
    req = urllib.request.Request(
        BASE + path,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    results["health"] = json.loads(get("/api/health"))
    print("health:", results["health"])

    up1 = upload_multipart(
        "/api/upload", files=[ROOT / "examples/example_bpe_large/tokenizer.json"],
        fields={"label": "bpe-800"},
    )
    up2 = upload_multipart(
        "/api/upload", files=[ROOT / "examples/example_bpe_small/tokenizer.json"],
        fields={"label": "bpe-300"},
    )
    up3 = upload_multipart(
        "/api/upload",
        files=[
            ROOT / "examples/example_bpe_vocab_merges/vocab.json",
            ROOT / "examples/example_bpe_vocab_merges/merges.txt",
        ],
        fields={"label": "bpe-vocab-merges"},
    )
    up4 = upload_multipart(
        "/api/upload", files=[ROOT / "examples/make_example_tokenizers.py"],
        fields={"label": "not-a-tokenizer"},
    )
    results["uploads"] = {"bpe800": up1, "bpe300": up2, "vocab_merges": up3, "invalid": up4}
    for name, up in results["uploads"].items():
        print(f"upload {name}: ok={up.get('ok')} "
              f"kind={(up.get('tokenizer') or {}).get('kind')} "
              f"vocab={(up.get('tokenizer') or {}).get('vocab_size')} err={up.get('error')}")

    sid_a = up1["tokenizer"]["session_id"]
    sid_b = up2["tokenizer"]["session_id"]

    pg_text = ("今天天气很好，我们一起去测试 tokenizer 的分词效果。\n"
               "The quick brown fox jumps over the lazy dog.")
    pg = post("/api/playground", {"session_id": sid_a, "text": pg_text})
    results["playground"] = pg
    print(f"playground: tokens={pg['count']} words={pg['word_count']} chars={pg['char_count']} "
          f"bytes={pg['byte_count']} fertility={pg['fertility']:.4f} "
          f"cr={pg['compression_ratio']:.4f} bc={pg['byte_compression']:.4f} "
          f"fidelity_pass={pg['fidelity_pass']}")
    print("  pieces:", [(t["surface"], t["id"]) for t in pg["tokens"][:10]])
    print("  decoded == input:", pg["decoded"].strip() == pg_text.strip())

    results["playground_special"] = post(
        "/api/playground",
        {"session_id": sid_a, "text": "hello world", "include_special_tokens": True},
    )
    print("playground(include_special):",
          [t["token"] for t in results["playground_special"]["tokens"]])

    ev = post("/api/evaluate", {"session_id": sid_a})
    results["evaluate_bpe800"] = ev
    s = ev["summary"]
    print(f"evaluate bpe-800: run={ev['run_id']} groups={len(ev['rows'])} "
          f"samples={s['samples_evaluated']}")
    print(f"  fertility={s['fertility']:.4f} compression={s['compression_ratio']:.4f} "
          f"byte_compression={s['byte_compression']:.4f} fidelity={s['fidelity_pass_rate']:.4f} "
          f"fails={s['fidelity_fails']}")
    print(f"  shannon={s['shannon_entropy']:.4f} renyi_0.5={s['renyi_entropy_alpha_0_5']:.4f} "
          f"renyi_1={s['renyi_entropy_alpha_1']:.4f} renyi_2={s['renyi_entropy_alpha_2']:.4f} "
          f"gini={s['fairness_gini']:.4f} score={s['composite_score']}")
    print(f"  avg_token_bytes={s['avg_token_bytes']:.4f} avg_token_chars={s['avg_token_chars']:.4f} "
          f"TTR={s['type_token_ratio']:.4f} vocab_util={s['vocabulary_utilization']:.4f} "
          f"unique={s['unique_tokens']} subword_fertility={s['subword_fertility']:.4f} "
          f"continued_word_rate={s['continued_word_rate']:.4f}")
    print("  per-language token cost:",
          json.dumps(s["per_language_token_cost"], ensure_ascii=False))
    for row in ev["rows"]:
        print(f"    {row['name']:<38} n={row['sample_count']:<3} avg_tok={row['avg_tokens']:>8.2f} "
              f"fert={row['fertility']:>7.3f} cr={row['compression_ratio']:.4f} "
              f"bc={row['byte_compression']:.4f} fails={row['fidelity_fails']} "
              f"grade={row['fertility_grade']}")

    ev2 = post("/api/evaluate", {"session_id": sid_b})
    results["evaluate_bpe300"] = ev2
    s3 = ev2["summary"]
    print(f"evaluate bpe-300: fertility={s3['fertility']:.4f} "
          f"byte_compression={s3['byte_compression']:.4f} fidelity={s3['fidelity_pass_rate']:.4f} "
          f"gini={s3['fairness_gini']:.4f} score={s3['composite_score']}")

    cmp_ = post("/api/compare", {"session_a": sid_a, "session_b": sid_b, "include_custom": True})
    results["compare"] = cmp_
    print("compare run:", cmp_["run_id"])
    for row in cmp_["leaderboard"]:
        print(f"  {row['label']:<10} score={row['composite_score']:<7} "
              f"fertility={row['fertility']:<8} bc={row['byte_compression']:<8} "
              f"fidelity={row['fidelity_pass_rate']} gini={row['fairness_gini']} "
              f"worst={row['worst_subcategory']}")
    print("  grouped groups:", len(cmp_["charts"]["grouped_bar"]["labels"]),
          "| languages:", cmp_["charts"]["language_bar"]["labels"])

    ev_csv = get(f"/api/export/{ev['run_id']}?format=csv")
    ev_json = get(f"/api/export/{ev['run_id']}?format=json")
    cmp_csv = get(f"/api/export/{cmp_['run_id']}?format=csv")
    cmp_json = get(f"/api/export/{cmp_['run_id']}?format=json")
    (OUT_DIR / "export_evaluate.csv").write_text(ev_csv, encoding="utf-8")
    (OUT_DIR / "export_evaluate.json").write_text(ev_json, encoding="utf-8")
    (OUT_DIR / "export_compare.csv").write_text(cmp_csv, encoding="utf-8")
    (OUT_DIR / "export_compare.json").write_text(cmp_json, encoding="utf-8")
    print("export: evaluate csv rows =", len(ev_csv.strip().splitlines()) - 1,
          "| compare csv rows =", len(cmp_csv.strip().splitlines()) - 1)

    errs = {}
    try:
        post("/api/evaluate", {"session_id": "nope"})
        errs["unknown_session"] = "no error raised"
    except urllib.error.HTTPError as exc:
        errs["unknown_session"] = f"HTTP {exc.code}: {exc.read().decode('utf-8')[:200]}"
    results["errors"] = errs
    print("error paths:", json.dumps(errs, ensure_ascii=False))

    custom_path = OUT_DIR / "custom_sample.txt"
    custom_path.write_text(
        "自定义语料样本一：中文分词测试。\n"
        "Custom corpus sample two: tokenizer evaluation.\n",
        encoding="utf-8",
    )
    results["custom_corpus"] = upload_multipart(
        "/api/corpus/custom",
        files=[custom_path],
        fields={"category": "custom", "subcategory": "demo"},
        field_name="file",
    )
    print("custom corpus:", results["custom_corpus"])

    results["corpus"] = json.loads(get("/api/corpus"))
    print("corpus: builtin_files=%s builtin_samples=%s custom_files=%s custom_samples=%s" % (
        results["corpus"]["builtin_files"], results["corpus"]["builtin_samples"],
        results["corpus"]["custom_files"], results["corpus"]["custom_samples"]))
    results["sessions"] = json.loads(get("/api/sessions"))

    (OUT_DIR / "e2e_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved:", OUT_DIR / "e2e_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
