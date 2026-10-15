---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 04526fdd866a8c5c9aceee6b83ed2c05_e8e8fba6c84711f1ac4552540064ee0f
    ReservedCode1: DrFMRjeFvz70+PUTvLkny6BaKbqPBu5XOESLUDf6iMCq87Jsv4qOW4JbygQVBAi92o6qGjAFzqBBUWSLlt14S5NoI4TZ8T9qiDUvZcYxXXRqC+ExDrNnOfRpCbyMfFxNojBnLQ3tOZQm1AHT1h6HStJZVr2eIJNlCY+aD8iOGzQKOAqtm/Y8DhK6tLI=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 04526fdd866a8c5c9aceee6b83ed2c05_e8e8fba6c84711f1ac4552540064ee0f
    ReservedCode2: DrFMRjeFvz70+PUTvLkny6BaKbqPBu5XOESLUDf6iMCq87Jsv4qOW4JbygQVBAi92o6qGjAFzqBBUWSLlt14S5NoI4TZ8T9qiDUvZcYxXXRqC+ExDrNnOfRpCbyMfFxNojBnLQ3tOZQm1AHT1h6HStJZVr2eIJNlCY+aD8iOGzQKOAqtm/Y8DhK6tLI=
---

# tok_eval — HuggingFace Fast Tokenizer 多维度评测平台

一个**真实可运行、零构建步骤**的 tokenizer 评测 Web 项目：

- 后端：Python 3 + FastAPI + `tokenizers` / `transformers`
- 前端：原生 HTML / CSS / JS + Chart.js（CDN，无需 npm / webpack）
- 一键启动，单页面三 Tab：**Playground / Evaluate / Compare**

评测维度参考 `kitefishai-tokenizerbench.hf.space`、`KiteFishAI/TokenizerBench`、
`AntonioLopardo/tokenizer-analysis-suite`、`bgub/tokka-bench` 等公开方案的指标设计，
覆盖基础层（fertility / compression / fidelity）、词表层（subword fertility / TTR / vocab utilization）
与进阶层（Shannon 熵 / Rényi 熵 / 跨语言 fairness Gini）。

---

## 1. 目录结构

```
D:\tok_eval
├─ app\
│  ├─ main.py              FastAPI 应用与全部 API 路由、结果落盘
│  ├─ tokenizer_loader.py  强制 use_fast=True 的 tokenizer 加载器（4 种策略）
│  ├─ metrics.py           全部评测指标（fertility / 熵 / Gini / fidelity …）
│  ├─ corpus.py            内置 + 自定义语料加载、manifest 生成
│  └─ schemas.py           Pydantic 请求 / 响应模型
├─ web\
│  ├─ index.html           单页前端（三 Tab）
│  ├─ style.css            深色主题样式
│  ├─ app.js               交互逻辑 + Chart.js 图表
│  ├─ favicon.ico          tok_eval 品牌图标（16 / 32 / 48 / 64 多尺寸）
│  └─ favicon.png          tok_eval 品牌图标（512，顶部 logo 与 PNG favicon）
├─ corpus\
│  ├─ builtin\<category>\<subcategory>.txt   内置语料（19 个子类 / 76 条样本）
│  ├─ custom\<category>\<subcategory>.<ext>  用户自定义语料
│  └─ manifest.json        语料清单与格式说明（启动/上传时自动刷新）
├─ uploads\<session>\      上传的 tokenizer 文件，按会话隔离
├─ results\                Evaluate / Compare 的运行结果（eval-*.json / cmp-*.json）
├─ examples\               测试 tokenizer / 品牌图标生成脚本
├─ tests\e2e_test.py       端到端 HTTP 冒烟测试
├─ requirements.txt
├─ run.bat
└─ README.md
```

---

## 2. 快速开始

```powershell
# 1) 安装依赖
cd D:\tok_eval
pip install -r requirements.txt

# 2) 启动服务（任选其一）
python -m uvicorn app.main:app --port 8000
run.bat

# 3) 打开浏览器
start http://127.0.0.1:8000
```

- 交互式 API 文档：`http://127.0.0.1:8000/docs`
- 依赖清单（`requirements.txt`）：`fastapi`、`uvicorn`、`python-multipart`、`transformers`、
  `tokenizers`、`numpy`、`pandas`，另有 `sentencepiece` + `protobuf` 用于 `*.model` 场景。

> **离线可用**：加载器内部默认设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，
> 评测过程**不会访问 huggingface.co**，全部计算在本机完成；
> 因此即使无法访问外网，也可以用本地构造的 tokenizer 完整跑通（见第 8 节）。

---

## 3. 语料规范

### 3.1 存放位置

```
corpus\builtin\<category>\<subcategory>.txt     内置语料
corpus\custom\<category>\<subcategory>.<ext>    用户自定义语料
```

### 3.2 文件格式（写入 `corpus/manifest.json` 的 `format` 字段）

| 扩展名 | 规则 |
| --- | --- |
| `.txt`  | UTF-8 纯文本，**一行一条样本**；空行忽略；`#` 开头的行视为注释 |
| `.jsonl`| 每行一个 JSON 对象，从 `text` 字段读取样本 |
| `.csv`  | 含 `text` 列，一行一条样本 |

### 3.3 内置类别与子类（19 个子类，每子类 4 条种子样本，共 76 条）

| Category | Subcategories |
| --- | --- |
| `human_languages` | english / hindi / chinese / arabic / japanese / german / russian / korean |
| `programming_languages` | python / javascript / sql / rust |
| `scientific_formulas` | algebra / calculus / physics / statistics |
| `edge_cases` | whitespace_control / long_tokens / mixed_scripts |

内置样本可直接编辑（保持"一行一条 + UTF-8"即可），刷新页面或重启服务后重新扫描。

### 3.4 用户自定义语料

- 方式一（前端 **Evaluate** Tab → “自定义语料”上传框）：上传 `.txt` / `.jsonl` / `.csv`，并填写
  `category` / `subcategory`，文件落盘到 `corpus\custom\<category>\<subcategory>.<ext>`。
- 方式二（API）：

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/corpus/custom `
  -F "file=@D:\tok_eval\tests\artifacts\custom_sample.txt" `
  -F "category=custom" -F "subcategory=demo"
# → {"ok":true,"saved_to":"D:\\tok_eval\\corpus\\custom\\custom\\demo.txt","samples":2}
```

`POST /api/evaluate` 默认把自定义语料一起评测（`include_custom=true`），也可只用内置语料。

---

## 4. 上传 tokenizer 的文件要求

上传目录按会话隔离：`uploads\<session_id>\`，`session_id` 形如 `20261015-110338-e5702194`，
**不同会话之间绝不串用**（同一 tokenizer 重复上传会得到新的 session）。

支持 4 种输入（自动识别，无需选择类型）：

| # | 输入 | 加载方式 | `kind` |
| --- | --- | --- | --- |
| 1 | `tokenizer.json` | `tokenizers.Tokenizer.from_file` | `tokenizer_json` |
| 2 | 完整 HF fast 目录：`tokenizer.json` + `tokenizer_config.json` + `vocab.json` + `merges.txt` + `special_tokens_map.json` + `*.model` | `AutoTokenizer.from_pretrained(dir, use_fast=True)`（`local_files_only`） | `hf_folder` |
| 3 | BPE 的 `vocab.json` + `merges.txt` | 手工组装 ByteLevel BPE | `bpe_vocab_merges` |
| 4 | SentencePiece 的 `*.model` | `SentencePieceUnigramTokenizer` / `SentencePieceBPETokenizer`（按 proto 的 `model_type` 自动选） | `sentencepiece` |

**fast 强制策略（硬性）**

- 每个分支都等价于 `use_fast=True`（`tokenizers.Tokenizer` 即 fast backend）。
- 若 `AutoTokenizer` 只能解析出 slow（Python）实现，加载器**不会静默接受**，而是抛出
  `NotFastTokenizerError`：

  > `AutoTokenizer 只解析出 slow (Python) tokenizer，本项目强制要求 fast tokenizer。`
  > `请上传 tokenizer.json 或使用支持 fast 实现的模型。`

  可在上传接口返回的 `error` 字段看到该提示。
- 无法识别的文件组合会明确报错，例如：

  > `无法识别的 tokenizer 文件组合：make_example_tokenizers.py。`
  > `支持：tokenizer.json；完整 HF fast 目录；vocab.json + merges.txt；*.model。`

响应示例：

```json
{
  "ok": true,
  "tokenizer": {
    "session_id": "20261015-110338-e5702194",
    "kind": "tokenizer_json",
    "is_fast": true,
    "vocab_size": 800,
    "files": ["tokenizer.json"],
    "warnings": [],
    "special_tokens": {}
  },
  "error": null
}
```

---

## 5. 前端三个 Tab

### Playground

输入任意文本（含中文 / emoji / 代码快捷键），实时返回：

- token 序列（按 token 高亮，可切换"是否计入 special tokens"）
- 每个 token 的 **Surface（覆盖的原文片段）**、**Raw token**、**Token ID**、`[start, end)` 字符边界、是否续接前词
- 单条文本的 `tokens / words / chars / bytes`、fertility、compression ratio、byte compression
- **encode → decode fidelity 校验**：解码结果与原文逐字符比对，给出通过/失败提示与差异对照

### Evaluate

对单个 tokenizer 在**全部内置语料类别 + 自定义语料**上全量评测，输出：

- 关键指标卡（fertility / compression / byte compression / fidelity / Shannon 熵 / Gini / composite score …）
- 分层结果表：`Category | Subcategory | Avg tokens | Fertility | Compression | Byte compression | Fidelity fails`
  （前端表格与 CSV 导出还带 grade、avg token bytes、TTR、vocab utilization、subword fertility、continued word rate、熵等列）
- 图表（Chart.js）：
  - 子分类 **Fertility 柱状图**
  - **语言 Fertility 对比**（human_languages 各语言）
  - **Fertility vs Compression 散点图**
  - 基础层 / 词表层 / 进阶层指标卡

### Compare

选择两个已上传 tokenizer 并排对比：

- **Leaderboard**：按 composite score 排名，展示 fertility、byte compression、fidelity、Gini、平均 token bytes、TTR、熵、`worst_subcategory` 等
- **分组柱状图**：每个子分类下两个 tokenizer 的 fertility 对照
- 语言维度对照柱状图

### 导出

每个 run 都落盘到 `results\<run_id>.json`，并可导出：

```
GET /api/export/{run_id}?format=csv      # 结果表（扁平 CSV）
GET /api/export/{run_id}?format=json     # 完整运行结果（含 charts）
GET /api/result/{run_id}                 # 重新读取已存储的 run
GET /api/runs                            # 列出所有已存储的 run
```

---

## 6. 评测指标定义

设某样本：`T` = token 数，`W` = 词数，`C` = 字符数，`B` = UTF-8 字节数。

### 6.1 基础层

| 指标 | 定义 | 方向 |
| --- | --- | --- |
| `avg_tokens` | `T` 的样本均值 | — |
| `fertility` | `T / W`（tokens per word） | 越接近 1 越好；**≥ 4 判定为差（bad）** |
| `compression_ratio` | `T / C`（tokens per char） | 越低越好 |
| `byte_compression` | `T / B`（tokens per UTF-8 byte） | 越低越好；跨语言公平比较的主口径 |
| `fidelity_pass_rate` | `encode → decode` 后与原文完全一致（含 NFD/NFC 归一化回退比较）的样本占比 | 越高越好，1.0 为理想 |
| `samples_evaluated` | 实际评测样本数 | — |

fertility 分级：`≤1.3 good` / `≤2 fair` / `≤4 poor` / `>4 bad`。

> **词切分约定**：CJK / 假名 / 谚文按**字**计数（每个字符算一个词），其余文字按空白切分，
> 纯标点片段被丢弃。这样中文、日文等无空格语言与英文的 fertility 才具备可比性。

### 6.2 词表层

| 指标 | 定义 |
| --- | --- |
| `avg_token_bytes` / `avg_token_chars` | 平均 token 长度（UTF-8 字节 / 字符） |
| `type_token_ratio` (TTR) | 唯一 token 数 / 总 token 数 |
| `vocabulary_utilization` | 唯一 token 数 / tokenizer 词表大小 |
| `unique_tokens` | 评测中出现的唯一 token 数 |
| `subword_fertility` | 每词平均被切出的"续接子词"数（词首 token 之外的 token 数 / W） |
| `continued_word_rate` | 被切分成多个 token 的词占比（0–1） |

### 6.3 进阶层

| 指标 | 定义 |
| --- | --- |
| `shannon_entropy` | token 频次分布的 Shannon 熵（bit）：`H = -Σ p_i log2 p_i` |
| `renyi_entropy_alpha_0_5 / 1 / 2` | Rényi 熵 `H_α = 1/(1-α) · log2(Σ p_i^α)`；α=1 退化为 Shannon 熵 |
| `fairness_gini` | 跨语言 **fairness Gini 系数**：语言 cost = `tokens / raw bytes`，**各语言等权**（先按语言算平均 cost，再对语言序列求 Gini） |
| `composite_score` | 综合分（0–100，越高越好）：由 byte compression、bbox fertility、fidelity、Gini 等归一化加权，用于 Compare 的 Leaderboard 排序 |

---

## 7. API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/` | 前端页面 |
| GET | `/api/health` | 存活状态 + 内置语料文件数 + 已加载会话数 |
| GET | `/api/sessions` | 本次服务运行中已上传的 tokenizer（含 label / kind / vocab_size / 文件列表） |
| GET | `/api/corpus` | 语料清单（内置 + 自定义，含样本数、类别子类、格式说明） |
| POST | `/api/corpus/custom` | 上传自定义语料（multipart：`file` + `category` + `subcategory`） |
| POST | `/api/upload` | 上传 tokenizer 文件（multipart：`files` 可多个 + `label`）→ `session_id` |
| POST | `/api/playground` | `{session_id, text, include_special_tokens}` → token 列表 + fidelity |
| POST | `/api/evaluate` | `{session_id, categories?, subcategories?, include_custom?, custom_only?}` → 分层结果 + 图表数据 |
| POST | `/api/compare` | `{session_a, session_b, categories?, include_custom?}` → Leaderboard + 分组图 |
| GET | `/api/result/{run_id}` | 重新读取已存储的 run |
| GET | `/api/export/{run_id}?format=csv\|json` | 导出结果 |

出错时返回标准 HTTP 状态码 + 明确中文说明，例如：

```json
{"detail": "session nope 不存在或尚未加载 tokenizer，请先上传。"}
```

---

## 8. 实测验证（已真实执行，非仅交付代码）

环境：Windows 11 · Python 3.11.8 · `tokenizers` / `transformers` 本地安装成功。

### 8.1 测试样本：本地构造的小型 fast tokenizer

`examples/make_example_tokenizers.py` 用 `tokenizers` 的 `BpeTrainer` 在**内置语料**（76 条）上训练，
产出两个 BPE tokenizer（**不依赖 huggingface.co**）：

| 样本 | 路径 | vocab |
| --- | --- | --- |
| tokenizer A | `examples\example_bpe_large\tokenizer.json` | 800 |
| tokenizer B | `examples\example_bpe_small\tokenizer.json` | 300 |
| 第三种输入 | `examples\example_bpe_vocab_merges\{vocab.json,merges.txt}` | 800 |

```powershell
python examples\make_example_tokenizers.py
# training samples: 76
# vocab small: 300
# vocab large: 800
```

### 8.2 端到端 HTTP 验证

服务启动后执行 `python tests\e2e_test.py`（脚本源码见 `tests/e2e_test.py`，
原始返回 JSON 见 `tests\artifacts\e2e_results.json`）：

```
health: {'status': 'ok', 'version': '1.0.0', 'builtin_corpus_files': 19, 'sessions': 0}
upload bpe800:        ok=True  kind=tokenizer_json    vocab=800
upload bpe300:        ok=True  kind=tokenizer_json    vocab=300
upload vocab_merges:  ok=True  kind=bpe_vocab_merges  vocab=800
upload invalid:       ok=False err=无法识别的 tokenizer 文件组合：make_example_tokenizers.py。…
playground: tokens=70 words=28 chars=76 bytes=116 fertility=2.5000 cr=0.9211 bc=0.6034 fidelity_pass=True
playground(include_special): ['h','el','lo','Ġw','or','l','d']
evaluate bpe-800: run=eval-20261015-110338-b39903 groups=20 samples=78
compare run: cmp-20261015-110338-cdce83
export: evaluate csv rows = 20 | compare csv rows = 40
custom corpus: {'ok': True, 'saved_to': 'D:\\tok_eval\\corpus\\custom\\custom\\demo.txt', 'samples': 2}
error paths: {"unknown_session": "HTTP 404: {\"detail\":\"session nope 不存在或尚未加载 tokenizer，请先上传。\"}"}
```

### 8.3 指标实测结果（tokenizer A，vocab 800，76 条内置 + 2 条自定义 = 78 条样本）

| 层 | 指标 | 值 |
| --- | --- | --- |
| 基础 | avg_tokens | 35.885 |
| 基础 | fertility (tokens/word) | **3.5702**（poor） |
| 基础 | compression_ratio (tokens/char) | 0.6672 |
| 基础 | byte_compression (tokens/byte) | **0.4963** |
| 基础 | fidelity_pass_rate | **1.0000**（0 失败） |
| 基础 | samples_evaluated | 78 |
| 词表 | avg_token_bytes / avg_token_chars | 2.6756 / 1.8260 |
| 词表 | type_token_ratio | 0.2301 |
| 词表 | vocabulary_utilization | 0.8050（unique 644 / 800） |
| 词表 | subword_fertility | 1.9987 |
| 词表 | continued_word_rate | 0.6454 |
| 进阶 | shannon_entropy | 8.8337 bit |
| 进阶 | renyi α=0.5 / 1 / 2 | 9.1017 / 8.8337 / 8.2035 |
| 进阶 | fairness_gini | 0.1183 |
| 进阶 | composite_score | 46.88 |

分层结果表（节选，完整 20 行见 CSV 导出）：

| Category | Subcategory | Avg tokens | Fertility | Compression | Byte compression | Fidelity fails |
| --- | --- | --- | --- | --- | --- | --- |
| human_languages | chinese | 41.00 | 2.050 | 1.9070 | 0.6357 | 0 |
| human_languages | english | 33.00 | 3.474 | 0.5019 | 0.5019 | 0 |
| human_languages | arabic | 39.75 | 4.184 | 0.7130 | 0.3897 | 0 |
| human_languages | hindi | 53.75 | 4.674 | 0.8811 | 0.3318 | 0 |
| human_languages | russian | 46.00 | 5.935 | 0.7764 | 0.4163 | 0 |
| programming_languages | python | 29.75 | 3.400 | 0.4630 | 0.4630 | 0 |
| programming_languages | rust | 36.25 | 5.577 | 0.5943 | 0.5943 | 0 |
| scientific_formulas | statistics | 31.25 | 6.250 | 0.5841 | 0.5841 | 0 |
| edge_cases | long_tokens | 25.25 | 25.250 | 0.3531 | 0.3531 | 0 |
| custom | demo | 30.00 | 3.000 | 0.9524 | 0.6316 | 0 |

各语言 token 成本（tokens / raw byte，用于 fairness Gini）：

```json
{"arabic":0.3897,"chinese":0.6357,"english":0.5019,"german":0.5956,
 "hindi":0.3318,"japanese":0.4932,"korean":0.6069,"russian":0.4163}
```

### 8.4 Compare 实测（A: vocab 800 vs B: vocab 300）

| Rank | Tokenizer | composite | fertility | byte compression | fidelity | Gini | worst_subcategory |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | bpe-800 | 46.88 | 3.5702 | 0.4963 | 1.0 | 0.1183 | edge_cases/long_tokens |
| 2 | bpe-300 | 41.11 | 5.9592 | 0.8284 | 1.0 | 0.0827 | edge_cases/long_tokens |

结论与直觉一致：词表更大 → token 更少（fertility / byte compression 明显更低），
但语言间差异略大（Gini 0.1183 > 0.0827），综合分仍由 bpe-800 领先。

### 8.5 其它已验证项

- 静态资源：`/` = 200、`/static/style.css` = 200、`/static/app.js` = 200、`/favicon.ico` = 200（`image/x-icon`，10148 B）、`/static/favicon.png` = 200（`image/png`，14193 B）、`/docs` = 200
- 导出：`/api/export/{run_id}?format=csv` 表头为
  `category,subcategory,name,sample_count,samples_evaluated,avg_tokens,fertility,fertility_grade,compression_ratio,byte_compression,fidelity_pa…`
- 会话隔离：`uploads\20261015-110338-e5702194\` 与其它会话目录互不影响
- 强制 fast：`python examples\verify_fast_enforcement.py` →

  ```
  PASS: slow-only tokenizer rejected -> AutoTokenizer 只解析出 slow (Python) tokenizer，
  本项目强制要求 fast tokenizer。请上传 tokenizer.json 或使用支持 fast 实现的模型。
  ```

### 8.6 关于 huggingface.co

本次验证**完全使用本地构造的 tokenizer**，未下载任何 HF 模型，故无需访问 huggingface.co。
若你的网络无法访问 huggingface.co，同样按第 8.1 节方式生成样本 tokenizer 即可完成全链路验证；
加载器本身已开启离线模式（`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`），
上传本地文件不会触发任何网络请求。

---

## 9. 品牌与图标（tok_eval）

前端品牌文案与图标已统一为 **tok_eval**：

| 位置 | 内容 |
|---|---|
| `<title>` | `tok_eval · HuggingFace Fast Tokenizer 评测` |
| 顶部 brand | `<img class="logo" src="/static/favicon.png">` + 站名 `<h1>tok_eval</h1>` |
| footer | `tok_eval v1.0 · FastAPI + 原生前端 + Chart.js` |
| 标签页图标 | `/favicon.ico`（多尺寸 ICO）+ `/static/favicon.png`（高清 PNG，兼作 apple-touch-icon） |
| API 文档 | `FastAPI(title="tok_eval")`，`/docs` 页面标题同步为 tok_eval |

图标由脚本生成，改配色或尺寸后可直接重新产出：

```bat
python examples\generate_brand_icon.py
```

脚本输出 `web\favicon.ico`（16 / 32 / 48 / 64 四种尺寸）与 `web\favicon.png`（512）。设计为深色圆角底 + 3×3 token 网格（中心 token 高亮），与页面 `--accent` / `--accent-2` 渐变同色系，在 16×16 下仍清晰可辨。

## 10. 常见问题

| 现象 | 说明 / 处理 |
| --- | --- |
| 上传后报 “只有 slow tokenizer” | 该文件集没有 fast 实现，请补充 `tokenizer.json`（`AutoTokenizer.save_pretrained` 时会生成） |
| `*.model` 上传失败 | 需要 `pip install sentencepiece protobuf` |
| GPT-2 风格 `vocab.json + merges.txt` 的 token 显示为 `Ġ` / `ä»` | ByteLevel BPE 的字节级表示，属正常；Playground 的 **Surface** 列显示对应原文片段 |
| 中文 fertility 偏小 | 中文按"字"计词（见 6.1 词切分约定），与英文按词计数不可直接横比，跨语言请用 `byte_compression` 与 Gini |
| `long_tokens` 子类 fertility 极高（如 25.25） | 该子类刻意放入超长无空格串（URL / 长 hash），整串只算 1 个词，是边界压力测试的预期表现 |
| 想改端口 | `python -m uvicorn app.main:app --port 8080` |
*（内容由AI生成，仅供参考）*
