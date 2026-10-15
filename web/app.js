/* tok_eval - frontend logic (vanilla JS + Chart.js) */
'use strict';

const state = {
  sessions: [],
  current: null,
  evalRun: null,
  cmpRun: null,
  categories: {},
  charts: {},
};

const SAMPLES = {
  '中文': '今天天气很好，我们一起去测试 tokenizer 的分词效果。\n北京大学的研究团队发布了新的大语言模型。',
  english: 'The quick brown fox jumps over the lazy dog. Tokenization quality matters a lot for LLM efficiency.',
  code: 'def evaluate(tokenizer, samples):\n    return [len(tokenizer.encode(s)) for s in samples]\n# TODO: add byte compression metric',
  math: '\\int_{0}^{\\infty} e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}',
  mixed: '混合文本 mixed script: Hello 世界 नमस्ते مرحبا こんにちは 한국어',
  emoji: 'Great job 🎉🚀 tokenizer! 中文表情 😀 测试 ✨ done',
};

/* ---------------------------------------------------------------- helpers */
const $ = (sel) => document.querySelector(sel);

async function api(path, options) {
  const res = await fetch(path, options);
  const text = await res.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; } catch (e) { data = { detail: text }; }
  if (!res.ok) {
    throw new Error((data && (data.error || data.detail)) || ('HTTP ' + res.status));
  }
  return data;
}

function showBox(el, message, kind) {
  el.className = 'result-box ' + (kind || '');
  el.textContent = message;
  el.classList.remove('hidden');
}

function fmt(v, digits) {
  const n = Number(v);
  if (!isFinite(n)) return '-';
  const d = digits === undefined ? 3 : digits;
  return n.toFixed(d).replace(/\.?0+$/, (m) => (m.indexOf('.') === 0 ? '' : m));
}

function gradeClass(v) {
  const n = Number(v);
  if (!isFinite(n) || n <= 0) return '';
  if (n < 2) return 'good';
  if (n < 4) return 'warn';
  return 'bad';
}

function cards(container, items) {
  container.innerHTML = items.map((it) => `
    <div class="mcard">
      <div class="k">${it.k}</div>
      <div class="v ${it.cls || ''}">${it.v}</div>
      ${it.note ? `<div class="k small">${it.note}</div>` : ''}
    </div>`).join('');
}

function hasChart() { return typeof Chart !== 'undefined'; }

function drawChart(id, config) {
  const el = document.getElementById(id);
  if (!el) return;
  if (state.charts[id]) { state.charts[id].destroy(); }
  if (!hasChart()) {
    el.parentElement.insertAdjacentHTML('beforeend',
      '<p class="small muted">Chart.js CDN 未能加载，图表不可用（数据仍可通过导出获得）。</p>');
    return;
  }
  state.charts[id] = new Chart(el.getContext('2d'), config);
}

const PALETTE = ['#4d8dff', '#26d0a0', '#ffb457', '#ff6b6b', '#b18cff', '#4ecdc4', '#ffa8d8', '#8ad06b'];

const BASE_OPTS = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: { legend: { labels: { color: '#93a1bd', boxWidth: 12 } } },
  scales: {
    x: { ticks: { color: '#93a1bd', maxRotation: 60, minRotation: 0 }, grid: { color: '#2c3852' } },
    y: { ticks: { color: '#93a1bd' }, grid: { color: '#2c3852' } },
  },
};

/* ---------------------------------------------------------------- upload */
function pickedFiles() {
  const a = Array.from($('#fileInput').files || []);
  const b = Array.from($('#dirInput').files || []);
  const seen = new Set();
  return a.concat(b).filter((f) => {
    const key = f.name;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function refreshPicked() {
  const files = pickedFiles();
  $('#pickedFiles').textContent = files.length
    ? `${files.length} 个文件：` + files.map((f) => f.name).join(', ')
    : '未选择文件';
}

async function doUpload() {
  const files = pickedFiles();
  if (!files.length) { alert('请先选择 tokenizer 文件或文件夹'); return; }
  const btn = $('#uploadBtn');
  btn.disabled = true;
  btn.textContent = '上传中…';
  try {
    const form = new FormData();
    files.forEach((f) => form.append('files', f, f.name));
    form.append('label', $('#labelInput').value || '');
    const data = await api('/api/upload', { method: 'POST', body: form });
    if (!data.ok) {
      showBox($('#uploadResult'), '加载失败：' + (data.error || ''), 'err');
    } else {
      state.current = data.tokenizer.session_id;
      const t = data.tokenizer;
      let msg = `加载成功\nsession: ${t.session_id}\n类型: ${t.kind}\nis_fast: ${t.is_fast}\n词表大小: ${t.vocab_size}\n文件: ${t.files.join(', ')}`;
      if (t.warnings && t.warnings.length) msg += '\n提示: ' + t.warnings.join(' | ');
      showBox($('#uploadResult'), msg, 'ok');
      await loadSessions();
    }
  } catch (err) {
    showBox($('#uploadResult'), '加载失败：' + err.message, 'err');
  } finally {
    btn.disabled = false;
    btn.textContent = '上传并加载';
  }
}

/* ---------------------------------------------------------------- sessions */
async function loadSessions() {
  const data = await api('/api/sessions');
  state.sessions = data.sessions || [];
  const opts = state.sessions.map((s) =>
    `<option value="${s.session_id}">${s.label} · ${s.kind} · vocab ${s.vocab_size}</option>`).join('');
  ['#pgSession', '#evSession', '#cmpA', '#cmpB'].forEach((id) => {
    const sel = $(id);
    const prev = sel.value;
    sel.innerHTML = opts || '<option value="">（请先上传 tokenizer）</option>';
    if (prev && state.sessions.some((s) => s.session_id === prev)) sel.value = prev;
  });
  if (state.sessions.length >= 2) {
    $('#cmpA').value = state.sessions[0].session_id;
    $('#cmpB').value = state.sessions[1].session_id;
  }
  if (state.current) { $('#pgSession').value = state.current; $('#evSession').value = state.current; }
  setButtons();
}

function setButtons() {
  const has = state.sessions.length > 0;
  $('#pgRun').disabled = !has;
  $('#evRun').disabled = !has;
  $('#cmpRun').disabled = state.sessions.length < 2;
}

/* ---------------------------------------------------------------- playground */
async function runPlayground() {
  const sid = $('#pgSession').value;
  if (!sid) { alert('请先上传 tokenizer'); return; }
  const btn = $('#pgRun');
  btn.disabled = true; btn.textContent = '处理中…';
  try {
    const data = await api('/api/playground', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sid,
        text: $('#pgText').value,
        include_special_tokens: $('#pgSpecial').checked,
      }),
    });
    renderPlayground(data);
  } catch (err) {
    showBox($('#pgFidelity'), '失败：' + err.message, 'err');
  } finally {
    btn.disabled = false; btn.textContent = 'Tokenize';
  }
}

function renderPlayground(d) {
  cards($('#pgCards'), [
    { k: 'Tokens', v: d.count },
    { k: 'Words', v: d.word_count },
    { k: 'Fertility (tok/word)', v: fmt(d.fertility, 3), cls: gradeClass(d.fertility) },
    { k: 'Compression (tok/char)', v: fmt(d.compression_ratio, 4) },
    { k: 'Byte compression (tok/byte)', v: fmt(d.byte_compression, 4) },
    { k: 'Chars / Bytes', v: `${d.char_count} / ${d.byte_count}` },
  ]);

  const escaped = (d.decoded || '').replace(/</g, '&lt;');
  const src = $('#pgText').value.replace(/</g, '&lt;');
  showBox(
    $('#pgFidelity'),
    (d.fidelity_pass ? '✅ encode → decode 保真通过（完全一致）' : '⚠️ encode → decode 保真失败（解码结果与原文不同）')
      + '\n原始: ' + src.slice(0, 300)
      + '\n解码: ' + escaped.slice(0, 300),
    d.fidelity_pass ? 'ok' : 'warn'
  );

  const stream = d.tokens.map((t) =>
    `<span class="tok ${t.is_continuation ? 'cont' : ''}" title="raw=${escapeHtml(t.token)} id=${t.id} span=[${t.start},${t.end})">`
    + `${escapeHtml(t.surface !== undefined && t.surface !== '' ? t.surface : t.token)}`
    + `<span class="tid">${t.id}</span></span>`).join('');
  const table = $('#pgTable');
  table.querySelector('thead').innerHTML =
    '<tr><th>#</th><th>Surface</th><th>Raw token</th><th>ID</th><th>Start</th><th>End</th><th>Continuation</th></tr>';
  table.querySelector('tbody').innerHTML = d.tokens.map((t) => `
    <tr>
      <td>${t.index}</td>
      <td class="token">${escapeHtml(t.surface)}</td>
      <td class="mono">${escapeHtml(t.token)}</td>
      <td class="mono">${t.id}</td>
      <td>${t.start}</td>
      <td>${t.end}</td>
      <td>${t.is_continuation ? 'yes' : 'no'}</td>
    </tr>`).join('');

  const old = $('#pgStream');
  if (old) old.remove();
  const streamBox = document.createElement('div');
  streamBox.id = 'pgStream';
  streamBox.className = 'token-stream';
  streamBox.innerHTML = stream;
  $('#pgFidelity').parentElement.insertBefore(streamBox, $('#pgTable').parentElement);
}

function escapeHtml(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/\u0120/g, 'Ġ').replace(/\u2581/g, '▁');
}

/* ---------------------------------------------------------------- corpus */
async function loadCorpus() {
  const data = await api('/api/corpus');
  state.categories = data.categories || {};
  const grid = $('#evCategories');
  grid.innerHTML = Object.keys(state.categories).map((cat) => {
    const subs = state.categories[cat];
    const id = 'cat-' + cat;
    return `<div class="cat-box">
      <h4><label><input type="checkbox" class="cat-all" data-cat="${id}" checked /> ${cat}</label></h4>
      ${subs.map((s) => `<label><input type="checkbox" class="sub-cat" data-cat="${id}" value="${s}" checked /> ${s}</label>`).join('')}
    </div>`;
  }).join('');
  grid.querySelectorAll('.cat-all').forEach((el) => {
    el.addEventListener('change', (e) => {
      grid.querySelectorAll(`.sub-cat[data-cat="${e.target.dataset.cat}"]`)
        .forEach((c) => { c.checked = e.target.checked; });
    });
  });
  $('#corpusInfo').innerHTML =
    `内置语料 <b>${data.builtin_files}</b> 个文件 / <b>${data.builtin_samples}</b> 条样本，自定义语料 <b>${data.custom_files}</b> 个文件 / <b>${data.custom_samples}</b> 条样本。<br>`
    + `目录：<code>${data.builtin_root}</code>（内置）、<code>${data.custom_root}</code>（自定义）；清单：<code>${data.corpus_root}\\manifest.json</code><br>`
    + `格式：.txt 一行一条（# 注释）· .jsonl 每行 {"text": "..."} · .csv 含 text 列`;
}

function selectedCategories() {
  const checked = Array.from(document.querySelectorAll('.sub-cat:checked'));
  if (!checked.length) return null;
  const byCat = {};
  checked.forEach((c) => {
    const cat = c.dataset.cat.replace('cat-', '');
    (byCat[cat] = byCat[cat] || []).push(c.value);
  });
  return byCat;
}

/* ---------------------------------------------------------------- evaluate */
async function runEvaluate() {
  const sid = $('#evSession').value;
  if (!sid) { alert('请先上传 tokenizer'); return; }
  const byCat = selectedCategories();
  if (!byCat) { alert('请至少选择一个语料子类'); return; }

  const btn = $('#evRun');
  btn.disabled = true; btn.textContent = '评测中…';
  showBox($('#evStatus'), '正在评测，请稍候…', 'warn');
  try {
    const data = await api('/api/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sid,
        categories: Object.keys(byCat),
        subcategories: Object.values(byCat).flat(),
        include_custom: $('#evCustom').checked,
      }),
    });
    if (!data.ok) { showBox($('#evStatus'), '失败：' + data.error, 'err'); return; }
    state.evalRun = data.run_id;
    renderEvaluate(data);
    showBox($('#evStatus'),
      `评测完成 · run_id=${data.run_id} · ${data.summary.samples_evaluated} 条样本 · ${data.rows.length} 个语料分组`, 'ok');
    $('#evCsv').disabled = false; $('#evJson').disabled = false;
  } catch (err) {
    showBox($('#evStatus'), '失败：' + err.message, 'err');
  } finally {
    btn.disabled = false; btn.textContent = '运行全量评测';
  }
}

function renderEvaluate(data) {
  const s = data.summary;
  cards($('#evCards'), [
    { k: 'Composite score', v: fmt(s.composite_score, 2), cls: s.composite_score > 60 ? 'good' : 'warn', note: '0-100，越高越好' },
    { k: 'Fertility (tok/word)', v: fmt(s.fertility, 3), cls: gradeClass(s.fertility), note: '越接近 1 越好，≥4 视为差' },
    { k: 'Compression (tok/char)', v: fmt(s.compression_ratio, 4) },
    { k: 'Byte compression', v: fmt(s.byte_compression, 4) },
    { k: 'Fidelity pass rate', v: fmt(s.fidelity_pass_rate * 100, 1) + '%', cls: s.fidelity_pass_rate > 0.99 ? 'good' : 'warn' },
    { k: 'Fidelity fails', v: s.fidelity_fails, cls: s.fidelity_fails === 0 ? 'good' : 'bad' },
    { k: 'Shannon entropy', v: fmt(s.shannon_entropy, 3), note: 'bits' },
    { k: 'Rényi α=0.5 / 1 / 2', v: `${fmt(s.renyi_entropy_alpha_0_5, 2)} / ${fmt(s.renyi_entropy_alpha_1, 2)} / ${fmt(s.renyi_entropy_alpha_2, 2)}` },
    { k: 'Fairness Gini', v: fmt(s.fairness_gini, 4), cls: s.fairness_gini < 0.15 ? 'good' : 'warn', note: '跨语言，越低越公平' },
    { k: 'Vocab utilization', v: fmt(s.vocabulary_utilization * 100, 1) + '%', note: `${s.unique_tokens} / ${s.vocab_size}` },
    { k: 'Avg token bytes', v: fmt(s.avg_token_bytes, 2) },
    { k: 'Continued word rate', v: fmt(s.continued_word_rate * 100, 1) + '%' },
  ]);

  $('#evTable tbody').innerHTML = data.rows.map((r) => `
    <tr>
      <td>${r.category_label || r.category}</td>
      <td>${r.subcategory}</td>
      <td>${fmt(r.avg_tokens, 2)}</td>
      <td>${fmt(r.fertility, 3)} <span class="badge ${gradeClass(r.fertility)}">${r.fertility_grade || ''}</span></td>
      <td>${fmt(r.compression_ratio, 4)}</td>
      <td>${fmt(r.byte_compression, 4)}</td>
      <td>${r.fidelity_fails}</td>
    </tr>`).join('');

  const full = $('#evFull');
  const cols = ['name', 'samples_evaluated', 'avg_tokens', 'fertility', 'compression_ratio', 'byte_compression',
    'fidelity_pass_rate', 'avg_token_bytes', 'avg_token_chars', 'type_token_ratio', 'vocabulary_utilization',
    'unique_tokens', 'subword_fertility', 'continued_word_rate', 'shannon_entropy',
    'renyi_entropy_alpha_0_5', 'renyi_entropy_alpha_1', 'renyi_entropy_alpha_2'];
  full.querySelector('thead').innerHTML = '<tr>' + cols.map((c) => `<th>${c}</th>`).join('') + '</tr>';
  full.querySelector('tbody').innerHTML = data.rows.map((r) =>
    '<tr>' + cols.map((c) => `<td>${typeof r[c] === 'number' ? fmt(r[c], 4) : (r[c] == null ? '' : r[c])}</td>`).join('') + '</tr>'
  ).join('');
  $('#evFullWrap').classList.remove('hidden');

  renderEvalCharts(data.charts);
}

function renderEvalCharts(charts) {
  const f = charts.fertility_by_subcategory;
  drawChart('chartFertility', {
    type: 'bar',
    data: {
      labels: f.full_labels,
      datasets: [{
        label: 'Fertility (tokens / word)',
        data: f.values,
        backgroundColor: f.values.map((v) => (v < 2 ? '#26d0a0' : v < 4 ? '#ffb457' : '#ff6b6b')),
      }],
    },
    options: Object.assign({}, BASE_OPTS, { plugins: { legend: { display: false } } }),
  });

  const l = charts.language_fertility;
  drawChart('chartLanguage', {
    type: 'bar',
    data: {
      labels: l.labels,
      datasets: [{ label: 'Fertility', data: l.values, backgroundColor: '#4d8dff' }],
    },
    options: Object.assign({}, BASE_OPTS, { plugins: { legend: { display: false } } }),
  });

  const p = charts.scatter.points;
  drawChart('chartScatter', {
    type: 'scatter',
    data: {
      datasets: [{
        label: 'subcategory',
        data: p.map((pt) => ({ x: pt.x, y: pt.y, name: pt.label })),
        backgroundColor: '#26d0a0',
        pointRadius: 6,
      }],
    },
    options: Object.assign({}, BASE_OPTS, {
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.raw.name}: fertility ${ctx.raw.y} @ compression ${ctx.raw.x}`,
          },
        },
      },
      scales: {
        x: { title: { display: true, text: 'compression ratio (tokens/char)', color: '#93a1bd' }, ticks: { color: '#93a1bd' }, grid: { color: '#2c3852' } },
        y: { title: { display: true, text: 'fertility (tokens/word)', color: '#93a1bd' }, ticks: { color: '#93a1bd' }, grid: { color: '#2c3852' } },
      },
    }),
  });
}

/* ---------------------------------------------------------------- compare */
async function runCompare() {
  const a = $('#cmpA').value;
  const b = $('#cmpB').value;
  if (!a || !b) { alert('需要上传至少两个 tokenizer'); return; }
  if (a === b) { alert('请选择两个不同的 tokenizer'); return; }

  const btn = $('#cmpRun');
  btn.disabled = true; btn.textContent = '对比中…';
  showBox($('#cmpStatus'), '正在对比评测，请稍候…', 'warn');
  try {
    const data = await api('/api/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_a: a,
        session_b: b,
        include_custom: $('#evCustom').checked,
      }),
    });
    if (!data.ok) { showBox($('#cmpStatus'), '失败：' + data.error, 'err'); return; }
    state.cmpRun = data.run_id;
    renderCompare(data);
    showBox($('#cmpStatus'), `对比完成 · run_id=${data.run_id} · ${data.label_a} vs ${data.label_b}`, 'ok');
    $('#cmpCsv').disabled = false; $('#cmpJson').disabled = false;
  } catch (err) {
    showBox($('#cmpStatus'), '失败：' + err.message, 'err');
  } finally {
    btn.disabled = false; btn.textContent = '对比评测';
  }
}

const LB_COLS = [
  ['composite_score', 'Score'],
  ['fertility', 'Fertility'],
  ['compression_ratio', 'Compression'],
  ['byte_compression', 'Byte compression'],
  ['fidelity_pass_rate', 'Fidelity'],
  ['fidelity_fails', 'Fidelity fails'],
  ['avg_token_bytes', 'Avg token bytes'],
  ['type_token_ratio', 'TTR'],
  ['vocabulary_utilization', 'Vocab util'],
  ['unique_tokens', 'Unique tokens'],
  ['subword_fertility', 'Subword fertility'],
  ['continued_word_rate', 'Continued word rate'],
  ['shannon_entropy', 'Shannon H'],
  ['renyi_entropy_alpha_0_5', 'Rényi α=0.5'],
  ['renyi_entropy_alpha_1', 'Rényi α=1'],
  ['renyi_entropy_alpha_2', 'Rényi α=2'],
  ['fairness_gini', 'Fairness Gini'],
  ['vocab_size', 'Vocab size'],
  ['samples_evaluated', 'Samples'],
];

function renderCompare(data) {
  const table = $('#cmpTable');
  table.querySelector('thead').innerHTML =
    '<tr><th>Rank</th><th>Tokenizer</th>' + LB_COLS.map((c) => `<th>${c[1]}</th>`).join('') + '</tr>';
  table.querySelector('tbody').innerHTML = data.leaderboard.map((r, i) => {
    const cells = LB_COLS.map(([k]) => {
      const v = r[k];
      if (k === 'fidelity_pass_rate') return `<td>${fmt(v * 100, 1)}%</td>`;
      if (k === 'vocabulary_utilization' || k === 'continued_word_rate') return `<td>${fmt(v * 100, 1)}%</td>`;
      if (typeof v === 'number') return `<td>${k === 'composite_score' ? '<b>' + fmt(v, 2) + '</b>' : fmt(v, 4)}</td>`;
      return `<td>${v == null ? '' : v}</td>`;
    }).join('');
    return `<tr><td>${i + 1}</td><td>${r.label}<br><span class="small muted">${r.session_id}</span></td>${cells}</tr>`;
  }).join('');
  if (data.leaderboard[0]) {
    table.querySelector('tbody').insertAdjacentHTML('beforeend',
      `<tr><td colspan="${LB_COLS.length + 2}" class="muted small">最差子类：${data.leaderboard.map((r) => `${r.label} → ${r.worst_subcategory}`).join(' ｜ ')}</td></tr>`);
  }

  const g = data.charts.grouped_bar;
  drawChart('chartCmpFertility', {
    type: 'bar',
    data: {
      labels: g.labels,
      datasets: [
        { label: g.label_a, data: g.series_a, backgroundColor: '#4d8dff' },
        { label: g.label_b, data: g.series_b, backgroundColor: '#26d0a0' },
      ],
    },
    options: BASE_OPTS,
  });

  const lb = data.charts.language_bar;
  drawChart('chartCmpLanguage', {
    type: 'bar',
    data: {
      labels: lb.labels,
      datasets: [
        { label: lb.label_a, data: lb.series_a, backgroundColor: '#4d8dff' },
        { label: lb.label_b, data: lb.series_b, backgroundColor: '#26d0a0' },
      ],
    },
    options: BASE_OPTS,
  });

  const sc = data.charts.scatter;
  drawChart('chartCmpScatter', {
    type: 'scatter',
    data: {
      datasets: [
        { label: sc.label_a, data: sc.points_a.map((p) => ({ x: p.x, y: p.y, name: p.label })), backgroundColor: '#4d8dff', pointRadius: 6 },
        { label: sc.label_b, data: sc.points_b.map((p) => ({ x: p.x, y: p.y, name: p.label })), backgroundColor: '#26d0a0', pointRadius: 6 },
      ],
    },
    options: Object.assign({}, BASE_OPTS, {
      scales: {
        x: { title: { display: true, text: 'compression ratio', color: '#93a1bd' }, ticks: { color: '#93a1bd' }, grid: { color: '#2c3852' } },
        y: { title: { display: true, text: 'fertility', color: '#93a1bd' }, ticks: { color: '#93a1bd' }, grid: { color: '#2c3852' } },
      },
    }),
  });
}

/* ---------------------------------------------------------------- boot */
function bind() {
  $('#uploadBtn').addEventListener('click', doUpload);
  $('#fileInput').addEventListener('change', refreshPicked);
  $('#dirInput').addEventListener('change', refreshPicked);

  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
      document.querySelectorAll('.panel').forEach((p) => p.classList.remove('active'));
      tab.classList.add('active');
      $('#panel-' + tab.dataset.tab).classList.add('active');
    });
  });

  document.querySelectorAll('.chip[data-sample]').forEach((chip) => {
    chip.addEventListener('click', () => { $('#pgText').value = SAMPLES[chip.dataset.sample] || ''; });
  });

  $('#pgRun').addEventListener('click', runPlayground);
  $('#evRun').addEventListener('click', runEvaluate);
  $('#cmpRun').addEventListener('click', runCompare);
  $('#evCsv').addEventListener('click', () => { if (state.evalRun) window.location = `/api/export/${state.evalRun}?format=csv`; });
  $('#evJson').addEventListener('click', () => { if (state.evalRun) window.location = `/api/export/${state.evalRun}?format=json`; });
  $('#cmpCsv').addEventListener('click', () => { if (state.cmpRun) window.location = `/api/export/${state.cmpRun}?format=csv`; });
  $('#cmpJson').addEventListener('click', () => { if (state.cmpRun) window.location = `/api/export/${state.cmpRun}?format=json`; });
}

async function boot() {
  bind();
  try {
    const h = await api('/api/health');
    const el = $('#health');
    el.textContent = `服务正常 v${h.version} · 内置语料 ${h.builtin_corpus_files} 个文件 · 已加载 ${h.sessions} 个 tokenizer`;
    el.className = 'status ok';
  } catch (err) {
    const el = $('#health');
    el.textContent = '服务异常：' + err.message;
    el.className = 'status err';
  }
  try { await loadSessions(); } catch (e) { /* ignore */ }
  try { await loadCorpus(); } catch (e) { /* ignore */ }
}

boot();
