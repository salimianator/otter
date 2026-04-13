#!/usr/bin/env python3
"""
app.py — OTTER Debugger
Flask app for interactively testing and visualising the OTTER pipeline.
Run with: python app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ── make src/ importable ────────────────────────────────────────────────────
SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC))

from flask import Flask, jsonify, request

from classifier import QueryClassifier
from encoder    import SentenceEncoder
from planner    import QueryPlanner
from segmenter  import SentenceSegmenter

app = Flask(__name__)

# ── initialise pipeline once at startup ─────────────────────────────────────
print("Initialising OTTER pipeline …")
segmenter  = SentenceSegmenter(min_words=6)
encoder    = SentenceEncoder()          # lazy-loads on first encode() call
classifier = QueryClassifier(encoder)  # encodes prototypes → triggers load
planner    = QueryPlanner()
print("Pipeline ready.")

# ── load example from qasper for the "Load Example" button ──────────────────
_EXAMPLE_DOC   = ""
_EXAMPLE_QUERY = ""
_QASPER = Path(__file__).parent / "data" / "longbench_qasper.jsonl"
if _QASPER.exists():
    with open(_QASPER) as _f:
        _rec = json.loads(_f.readline())
    # Use first 600 words of context so the example loads quickly
    _words = _rec["context"].split()
    _EXAMPLE_DOC   = " ".join(_words[:600])
    _EXAMPLE_QUERY = _rec["input"]


# ── routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return HTML_TEMPLATE, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/example")
def example():
    return jsonify({"document": _EXAMPLE_DOC, "query": _EXAMPLE_QUERY})


@app.route("/compress", methods=["POST"])
def compress():
    body      = request.get_json(force=True)
    document  = body.get("document", "").strip()
    query     = body.get("query", "").strip()
    threshold = float(body.get("threshold", 0.85))

    if not document or not query:
        return jsonify({"error": "document and query are required"}), 400

    # 1 — segment
    sentences = segmenter.segment(document)
    if not sentences:
        return jsonify({"error": "No sentences detected in document."}), 400

    # 2 — encode
    sent_vecs = encoder.encode(sentences)
    query_vec = encoder.encode_query(query)

    # 3 — classify
    weights = classifier.get_weights(query)

    # 4 — score
    score_d = planner.score(sent_vecs, query_vec, weights)
    combined = score_d["combined"]

    # 5 — select (returns kept sentence strings)
    kept_texts = planner.select(sentences, combined, threshold=threshold)
    kept_set   = set(kept_texts)  # fast membership; texts are unique enough

    # Build per-sentence payload
    # Use index-based kept tracking for robustness (duplicate texts)
    import numpy as np
    ranked_idx  = np.argsort(combined)[::-1]
    cumsum      = np.cumsum(combined[ranked_idx])
    total_score = float(cumsum[-1])
    cutoff_pos  = int(np.searchsorted(cumsum, threshold * total_score))
    cutoff_pos  = min(cutoff_pos, len(sentences) - 1)
    kept_indices = set(ranked_idx[: cutoff_pos + 1].tolist())

    sentence_rows = []
    for i, text in enumerate(sentences):
        sentence_rows.append({
            "index":         i,
            "text":          text,
            "anchor_score":  round(float(score_d["anchor"][i]),  4),
            "flow_score":    round(float(score_d["flow"][i]),    4),
            "flash_score":   round(float(score_d["flash"][i]),   4),
            "combined_score":round(float(combined[i]),           4),
            "kept":          i in kept_indices,
        })

    kept_count  = len(kept_indices)
    total_count = len(sentences)
    comp_ratio  = kept_count / total_count
    output_text = " ".join(sentences[i] for i in sorted(kept_indices))

    return jsonify({
        "query":      query,
        "classifier": {
            "extractive_score": weights["extractive_score"],
            "alpha":            weights["alpha"],
            "beta":             weights["beta"],
            "gamma":            weights["gamma"],
        },
        "sentences": sentence_rows,
        "stats": {
            "total_sentences":    total_count,
            "kept_sentences":     kept_count,
            "compression_ratio":  round(comp_ratio, 4),
            "token_reduction_pct":round((1 - comp_ratio) * 100, 2),
        },
        "output": output_text,
    })


# ── HTML template (self-contained, no external deps) ─────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OTTER Debugger</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #f0f4f8; color: #1a202c; min-height: 100vh; }

  /* ── header ── */
  header { background: #1a3a5c; color: #fff; padding: 18px 32px;
           display: flex; align-items: baseline; gap: 16px; }
  header h1 { font-size: 1.5rem; font-weight: 700; letter-spacing: .5px; }
  header p  { font-size: .85rem; opacity: .75; }

  /* ── layout ── */
  .container { max-width: 1400px; margin: 0 auto; padding: 24px 20px; }

  /* ── card ── */
  .card { background: #fff; border-radius: 10px;
          box-shadow: 0 1px 4px rgba(0,0,0,.1); padding: 20px; }

  /* ── input panel ── */
  #input-panel { margin-bottom: 20px; }
  #input-panel label { display: block; font-size: .8rem; font-weight: 600;
                       color: #4a5568; margin-bottom: 5px; text-transform: uppercase;
                       letter-spacing: .4px; }
  textarea, input[type=text] {
    width: 100%; border: 1px solid #cbd5e0; border-radius: 6px;
    padding: 10px 12px; font-size: .92rem; font-family: inherit;
    resize: vertical; outline: none; transition: border-color .2s;
  }
  textarea:focus, input[type=text]:focus { border-color: #4299e1; }
  textarea { height: 160px; }
  .row2 { display: grid; grid-template-columns: 1fr auto; gap: 16px;
           align-items: end; margin-top: 14px; }
  .query-wrap { display: grid; gap: 6px; }
  .slider-wrap { display: flex; flex-direction: column; gap: 6px; min-width: 220px; }
  .slider-row  { display: flex; align-items: center; gap: 10px; }
  input[type=range] { flex: 1; accent-color: #4299e1; }
  #thresh-val { font-weight: 700; color: #2b6cb0; font-size: .95rem; min-width: 36px; }
  .btn-row { display: flex; gap: 10px; margin-top: 14px; }
  button { cursor: pointer; border: none; border-radius: 6px; padding: 10px 22px;
           font-size: .92rem; font-weight: 600; transition: background .15s, transform .1s; }
  button:active { transform: scale(.97); }
  #btn-compress { background: #2b6cb0; color: #fff; }
  #btn-compress:hover { background: #2c5282; }
  #btn-example  { background: #e2e8f0; color: #2d3748; }
  #btn-example:hover { background: #cbd5e0; }

  /* ── spinner ── */
  #spinner { display: none; position: fixed; inset: 0;
             background: rgba(255,255,255,.65); z-index: 999;
             align-items: center; justify-content: center; flex-direction: column; gap: 16px; }
  #spinner.active { display: flex; }
  .spin-ring { width: 52px; height: 52px; border: 5px solid #bee3f8;
               border-top-color: #2b6cb0; border-radius: 50%;
               animation: spin .8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  #spinner p { color: #2b6cb0; font-weight: 600; font-size: .95rem; }

  /* ── results grid ── */
  #results { display: none; }
  #results.active { display: block; }
  .results-grid { display: grid;
                  grid-template-columns: 260px 1fr 280px 1fr;
                  gap: 16px; align-items: start; }
  @media (max-width: 1100px) {
    .results-grid { grid-template-columns: 1fr 1fr; }
  }
  @media (max-width: 640px) {
    .results-grid { grid-template-columns: 1fr; }
  }
  .section-title { font-size: .75rem; font-weight: 700; text-transform: uppercase;
                   letter-spacing: .5px; color: #718096; margin-bottom: 14px; }

  /* ── Section 1 — Classifier ── */
  .ext-bar-wrap { margin-bottom: 16px; }
  .ext-labels { display: flex; justify-content: space-between;
                font-size: .72rem; color: #718096; margin-bottom: 4px; }
  .ext-track { background: #e2e8f0; border-radius: 99px; height: 12px;
               overflow: hidden; position: relative; }
  .ext-fill  { height: 100%; background: linear-gradient(90deg,#667eea,#f6ad55);
               border-radius: 99px; transition: width .4s ease; }
  .ext-score-label { text-align: center; font-size: .8rem; color: #4a5568;
                     margin-top: 5px; }
  .badges { display: flex; flex-direction: column; gap: 8px; }
  .badge { display: flex; justify-content: space-between; align-items: center;
           background: #f7fafc; border-radius: 6px; padding: 7px 12px;
           border-left: 3px solid; }
  .badge-label { font-size: .8rem; font-weight: 600; color: #4a5568; }
  .badge-val   { font-size: .9rem; font-weight: 700; font-family: monospace; }
  .badge.alpha { border-color: #4299e1; }
  .badge.beta  { border-color: #ed8936; }
  .badge.gamma { border-color: #38b2ac; }

  /* ── Section 2 — Sentence list ── */
  #sent-list { max-height: 480px; overflow-y: auto; display: flex;
               flex-direction: column; gap: 4px; }
  .sent-row { padding: 7px 10px; border-radius: 6px; display: grid;
              grid-template-columns: 28px 1fr 90px 28px; gap: 8px;
              align-items: center; font-size: .82rem; }
  .sent-row.kept    { background: #f0fff4; }
  .sent-row.dropped { background: #f7fafc; }
  .sent-idx  { color: #a0aec0; font-size: .72rem; font-family: monospace; text-align:right; }
  .sent-text { white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
               color: #2d3748; }
  .sent-row.dropped .sent-text { color: #a0aec0; }
  .score-bar-wrap { position: relative; height: 8px; background: #e2e8f0;
                    border-radius: 99px; overflow: hidden; }
  .score-bar-fill { position: absolute; left: 0; top: 0; height: 100%;
                    border-radius: 99px; transition: width .3s; }
  .kept    .score-bar-fill { background: #48bb78; }
  .dropped .score-bar-fill { background: #a0aec0; }
  .score-num { font-size: .75rem; font-family: monospace; color: #718096;
               text-align: right; }
  .sent-icon { font-size: .9rem; text-align: center; }

  /* ── Section 3 — SVG chart ── */
  #chart-wrap { overflow-x: auto; }
  #chart-wrap svg text { font-family: -apple-system, sans-serif; }

  /* ── Section 4 — Output ── */
  #output-box { background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 8px;
                padding: 14px; font-size: .88rem; line-height: 1.6; color: #2d3748;
                max-height: 360px; overflow-y: auto; white-space: pre-wrap;
                word-break: break-word; margin-bottom: 14px; }
  .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
                margin-bottom: 14px; }
  .stat-box  { background: #ebf8ff; border-radius: 8px; padding: 10px 14px;
               text-align: center; }
  .stat-val  { font-size: 1.2rem; font-weight: 700; color: #2b6cb0; }
  .stat-lbl  { font-size: .7rem; color: #718096; text-transform: uppercase;
               letter-spacing: .4px; margin-top: 2px; }
  #btn-copy  { background: #e2e8f0; color: #2d3748; font-size: .85rem;
               padding: 8px 16px; }
  #btn-copy:hover { background: #cbd5e0; }
</style>
</head>
<body>

<header>
  <h1>OTTER Debugger</h1>
  <p>Sentence-Aware Prompt Compression</p>
</header>

<div id="spinner">
  <div class="spin-ring"></div>
  <p>Running OTTER pipeline…</p>
</div>

<div class="container">

  <!-- Input panel -->
  <div class="card" id="input-panel">
    <label for="doc-input">Document</label>
    <textarea id="doc-input" placeholder="Paste your document here…"></textarea>

    <div class="row2">
      <div class="query-wrap">
        <label for="query-input">Query</label>
        <input type="text" id="query-input" placeholder="Enter your query…">
      </div>
      <div class="slider-wrap">
        <label>Threshold</label>
        <div class="slider-row">
          <input type="range" id="threshold" min="0.5" max="0.99" step="0.01" value="0.85">
          <span id="thresh-val">0.85</span>
        </div>
      </div>
    </div>

    <div class="btn-row">
      <button id="btn-compress">&#9654; Compress</button>
      <button id="btn-example">Load Example</button>
    </div>
  </div>

  <!-- Results -->
  <div id="results">
    <div class="results-grid">

      <!-- 1. Classifier -->
      <div class="card">
        <div class="section-title">Classifier</div>
        <div class="ext-bar-wrap">
          <div class="ext-labels"><span>Abstractive</span><span>Extractive</span></div>
          <div class="ext-track"><div class="ext-fill" id="ext-fill" style="width:50%"></div></div>
          <div class="ext-score-label" id="ext-score-lbl">score: 0.50</div>
        </div>
        <div class="badges">
          <div class="badge alpha">
            <span class="badge-label">&#593; Alpha (Anchor)</span>
            <span class="badge-val" id="w-alpha">—</span>
          </div>
          <div class="badge beta">
            <span class="badge-label">&#946; Beta (Flow)</span>
            <span class="badge-val" id="w-beta">—</span>
          </div>
          <div class="badge gamma">
            <span class="badge-label">&#947; Gamma (Flash)</span>
            <span class="badge-val" id="w-gamma">—</span>
          </div>
        </div>
      </div>

      <!-- 2. Sentence scores -->
      <div class="card">
        <div class="section-title">Sentence Scores <span id="sent-count-lbl" style="font-weight:400;text-transform:none;letter-spacing:0;color:#a0aec0"></span></div>
        <div id="sent-list"></div>
      </div>

      <!-- 3. Score breakdown chart -->
      <div class="card">
        <div class="section-title">Score Breakdown (top 20)</div>
        <div id="chart-wrap"></div>
        <div style="display:flex;gap:12px;margin-top:10px;flex-wrap:wrap;">
          <span style="font-size:.75rem;display:flex;align-items:center;gap:4px;">
            <span style="display:inline-block;width:12px;height:12px;background:#4299e1;border-radius:2px"></span>Anchor</span>
          <span style="font-size:.75rem;display:flex;align-items:center;gap:4px;">
            <span style="display:inline-block;width:12px;height:12px;background:#ed8936;border-radius:2px"></span>Flow</span>
          <span style="font-size:.75rem;display:flex;align-items:center;gap:4px;">
            <span style="display:inline-block;width:12px;height:12px;background:#38b2ac;border-radius:2px"></span>Flash</span>
        </div>
      </div>

      <!-- 4. Output -->
      <div class="card">
        <div class="section-title">Output</div>
        <div id="output-box"></div>
        <div class="stats-grid">
          <div class="stat-box">
            <div class="stat-val" id="st-total">—</div>
            <div class="stat-lbl">Total sents</div>
          </div>
          <div class="stat-box">
            <div class="stat-val" id="st-kept">—</div>
            <div class="stat-lbl">Kept sents</div>
          </div>
          <div class="stat-box">
            <div class="stat-val" id="st-ratio">—</div>
            <div class="stat-lbl">Kept ratio</div>
          </div>
          <div class="stat-box">
            <div class="stat-val" id="st-reduction">—</div>
            <div class="stat-lbl">Token reduction</div>
          </div>
        </div>
        <button id="btn-copy">&#128203; Copy to clipboard</button>
      </div>

    </div><!-- .results-grid -->
  </div><!-- #results -->

</div><!-- .container -->

<script>
// ── threshold slider live readout ────────────────────────────────────────────
const slider   = document.getElementById('threshold');
const threshLbl= document.getElementById('thresh-val');
slider.addEventListener('input', () => { threshLbl.textContent = slider.value; });

// ── Load Example ─────────────────────────────────────────────────────────────
document.getElementById('btn-example').addEventListener('click', async () => {
  const res  = await fetch('/example');
  const data = await res.json();
  document.getElementById('doc-input').value   = data.document;
  document.getElementById('query-input').value = data.query;
});

// ── Compress ─────────────────────────────────────────────────────────────────
document.getElementById('btn-compress').addEventListener('click', async () => {
  const doc   = document.getElementById('doc-input').value.trim();
  const query = document.getElementById('query-input').value.trim();
  const thresh= parseFloat(slider.value);

  if (!doc || !query) { alert('Please enter both a document and a query.'); return; }

  // show spinner
  document.getElementById('spinner').classList.add('active');
  document.getElementById('results').classList.remove('active');

  try {
    const res  = await fetch('/compress', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ document: doc, query, threshold: thresh }),
    });
    const data = await res.json();
    if (data.error) { alert('Error: ' + data.error); return; }
    renderResults(data);
  } catch(e) {
    alert('Request failed: ' + e.message);
  } finally {
    document.getElementById('spinner').classList.remove('active');
  }
});

// ── render ───────────────────────────────────────────────────────────────────
function renderResults(data) {
  // 1. Classifier
  const ext = data.classifier.extractive_score;
  document.getElementById('ext-fill').style.width = (ext * 100).toFixed(1) + '%';
  document.getElementById('ext-score-lbl').textContent =
    'extractive score: ' + ext.toFixed(4) + (ext >= 0.55 ? '  (extractive)' : ext <= 0.45 ? '  (abstractive)' : '  (mixed)');
  document.getElementById('w-alpha').textContent = data.classifier.alpha.toFixed(4);
  document.getElementById('w-beta').textContent  = data.classifier.beta.toFixed(4);
  document.getElementById('w-gamma').textContent = data.classifier.gamma.toFixed(4);

  // 2. Sentence list
  const sentences = data.sentences;
  const maxCombined = Math.max(...sentences.map(s => s.combined_score));
  const listEl = document.getElementById('sent-list');
  listEl.innerHTML = '';
  sentences.forEach(s => {
    const pct = maxCombined > 0 ? (s.combined_score / maxCombined * 100).toFixed(1) : '0';
    const short = s.text.length > 80 ? s.text.slice(0, 79) + '…' : s.text;
    const row = document.createElement('div');
    row.className = 'sent-row ' + (s.kept ? 'kept' : 'dropped');
    row.title = s.text;
    row.innerHTML =
      '<span class="sent-idx">' + s.index + '</span>' +
      '<span class="sent-text">' + escHtml(short) + '</span>' +
      '<div class="score-bar-wrap"><div class="score-bar-fill" style="width:' + pct + '%"></div></div>' +
      '<span class="score-num">' + s.combined_score.toFixed(2) + '</span>' +
      '<span class="sent-icon">' + (s.kept ? '✅' : '❌') + '</span>';
    listEl.appendChild(row);
  });
  const keptN = sentences.filter(s => s.kept).length;
  document.getElementById('sent-count-lbl').textContent =
    '(' + keptN + ' / ' + sentences.length + ')';

  // 3. SVG chart — top 20 by combined score
  renderChart(sentences);

  // 4. Output
  document.getElementById('output-box').textContent = data.output;
  document.getElementById('st-total').textContent     = data.stats.total_sentences;
  document.getElementById('st-kept').textContent      = data.stats.kept_sentences;
  document.getElementById('st-ratio').textContent     = (data.stats.compression_ratio * 100).toFixed(1) + '%';
  document.getElementById('st-reduction').textContent = data.stats.token_reduction_pct.toFixed(1) + '%';

  document.getElementById('results').classList.add('active');
  document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderChart(sentences) {
  // pick top 20 by combined, keep doc order for display
  const top20 = [...sentences]
    .sort((a, b) => b.combined_score - a.combined_score)
    .slice(0, 20)
    .sort((a, b) => a.index - b.index);

  const W       = 220;   // bar area width (px)
  const rowH    = 13;    // height per bar
  const gap     = 3;
  const labelW  = 30;    // left label width
  const padT    = 6;
  const padB    = 20;
  const chartH  = top20.length * (rowH + gap) + padT + padB;
  const totalW  = labelW + W + 4;

  const maxScore = Math.max(...top20.map(s => s.combined_score)) || 1;

  let bars = '';
  top20.forEach((s, i) => {
    const y  = padT + i * (rowH + gap);
    const aw = (s.anchor_score / maxScore * W).toFixed(1);
    const fw = (s.flow_score   / maxScore * W).toFixed(1);
    const flw= (s.flash_score  / maxScore * W).toFixed(1);
    // stacked: anchor | flow | flash
    const x0 = labelW;
    const x1 = labelW + parseFloat(aw);
    const x2 = x1     + parseFloat(fw);
    const kept = s.kept ? '' : 'opacity=".45"';

    bars += `
      <text x="${labelW - 4}" y="${y + rowH - 2}" text-anchor="end"
            font-size="9" fill="#718096">${s.index}</text>
      <rect x="${x0}" y="${y}" width="${aw}" height="${rowH}"
            fill="#4299e1" rx="2" ${kept}/>
      <rect x="${x1}" y="${y}" width="${fw}" height="${rowH}"
            fill="#ed8936" rx="2" ${kept}/>
      <rect x="${x2}" y="${y}" width="${flw}" height="${rowH}"
            fill="#38b2ac" rx="2" ${kept}/>`;
  });

  // x-axis ticks
  const ticks = [0, 0.5, 1].map(t => {
    const x = labelW + t * W;
    const lbl = (t * maxScore).toFixed(2);
    return `<line x1="${x}" y1="${padT}" x2="${x}" y2="${chartH - padB}"
                  stroke="#e2e8f0" stroke-width="1"/>
            <text x="${x}" y="${chartH - padB + 12}" text-anchor="middle"
                  font-size="8" fill="#a0aec0">${lbl}</text>`;
  }).join('');

  document.getElementById('chart-wrap').innerHTML =
    `<svg width="${totalW}" height="${chartH}" style="display:block;overflow:visible">
      ${ticks}${bars}
    </svg>`;
}

// ── copy to clipboard ─────────────────────────────────────────────────────────
document.getElementById('btn-copy').addEventListener('click', () => {
  const txt = document.getElementById('output-box').textContent;
  navigator.clipboard.writeText(txt).then(() => {
    const btn = document.getElementById('btn-copy');
    btn.textContent = '✓ Copied!';
    setTimeout(() => { btn.textContent = '📋 Copy to clipboard'; }, 1800);
  });
});

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
</script>
</body>
</html>"""


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
