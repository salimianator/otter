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

from compress import OTTERCompressor

app = Flask(__name__)

# ── initialise pipeline once at startup ─────────────────────────────────────
print("Initialising OTTER pipeline …")
compressor = OTTERCompressor()
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
    import numpy as np

    body     = request.get_json(force=True)
    document = body.get("document", "").strip()
    query    = body.get("query", "").strip()

    if not document:
        return jsonify({"error": "A document is required."}), 400

    # Run the full OTTER pipeline via OTTERCompressor
    result = compressor.compress(document, query)

    if result["original_sentences"] == 0:
        return jsonify({"error": "No sentences detected in document."}), 400

    # Unpack scores for the per-sentence visualisation payload
    sentences  = compressor.segmenter.segment(document)   # re-segment for row data
    score_d    = result["scores"]
    combined   = score_d["combined"]
    weights    = result["weights"]

    # Determine kept indices: re-run select() to get the same boolean mask
    # (select() is deterministic, so this is safe and avoids storing extra state)
    _kept_texts  = set(result["compressed"].split())        # not reliable
    # Instead: rank by combined score and mark top-N where N = kept_sentences
    n_kept       = result["kept_sentences"]
    ranked_idx   = np.argsort(combined)[::-1]
    kept_indices = set(ranked_idx[:n_kept].tolist())

    sentence_rows = [
        {
            "index":          i,
            "text":           text,
            "anchor_score":   round(float(score_d["anchor"][i]),  4),
            "flow_score":     round(float(score_d["flow"][i]),    4),
            "flash_score":    round(float(score_d["flash"][i]),   4),
            "combined_score": round(float(combined[i]),           4),
            "kept":           i in kept_indices,
        }
        for i, text in enumerate(sentences)
    ]

    return jsonify({
        "query":      query,
        "classifier": {
            "w_ext":    weights["w_ext"],
            "w_enum":   weights["w_enum"],
            "w_abs":    weights["w_abs"],
            "dominant": weights["dominant"],
            "alpha":    weights["alpha"],
            "beta":     weights["beta"],
            "gamma":    weights["gamma"],
        },
        "selection_method": result["selection_method"],
        "sentences": sentence_rows,
        "stats": {
            "total_sentences":    result["original_sentences"],
            "kept_sentences":     result["kept_sentences"],
            "compression_ratio":  round(result["compression_ratio"], 4),
            "token_reduction_pct":round(result["token_reduction_pct"], 2),
        },
        "output": result["compressed"],
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
  .method-wrap { display: flex; flex-direction: column; gap: 6px; min-width: 200px; }
  .method-badge { display: inline-flex; align-items: center; gap: 8px;
                  background: #ebf8ff; border: 1px solid #bee3f8; border-radius: 8px;
                  padding: 9px 14px; font-size: .88rem; font-weight: 600; color: #2b6cb0; }
  .method-badge .dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  .method-badge.kneedle       .dot { background: #38b2ac; }
  .method-badge.marginal      .dot { background: #ed8936; }
  .method-badge.pending       .dot { background: #a0aec0; }
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
  .tri-bar-wrap { margin-bottom: 16px; }
  .tri-labels { display: flex; justify-content: space-between;
                font-size: .72rem; color: #718096; margin-bottom: 4px; }
  .tri-track { background: #e2e8f0; border-radius: 99px; height: 12px;
               overflow: hidden; display: flex; }
  .tri-seg   { height: 100%; transition: width .4s ease; }
  .tri-seg.ext  { background: #38b2ac; }
  .tri-seg.enum { background: #667eea; }
  .tri-seg.abs  { background: #f6ad55; }
  .tri-dominant { text-align: center; font-size: .8rem; font-weight: 600;
                  color: #4a5568; margin-top: 5px; }
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
      <div class="method-wrap">
        <label>Selection Method</label>
        <div class="method-badge pending" id="method-badge">
          <span class="dot"></span>
          <span id="method-label">Run pipeline to see</span>
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
        <div class="tri-bar-wrap">
          <div class="tri-labels">
            <span>Extractive</span><span>Enumeration</span><span>Abstractive</span>
          </div>
          <div class="tri-track">
            <div class="tri-seg ext"  id="tri-ext"  style="width:33%"></div>
            <div class="tri-seg enum" id="tri-enum" style="width:33%"></div>
            <div class="tri-seg abs"  id="tri-abs"  style="width:34%"></div>
          </div>
          <div class="tri-dominant" id="tri-dominant">—</div>
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

  if (!doc) { alert('Please enter a document.'); return; }

  // show spinner
  document.getElementById('spinner').classList.add('active');
  document.getElementById('results').classList.remove('active');

  try {
    const res  = await fetch('/compress', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ document: doc, query }),
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
  // 1. Classifier — three-class softmax bar
  const cl = data.classifier;
  document.getElementById('tri-ext').style.width  = (cl.w_ext  * 100).toFixed(1) + '%';
  document.getElementById('tri-enum').style.width = (cl.w_enum * 100).toFixed(1) + '%';
  document.getElementById('tri-abs').style.width  = (cl.w_abs  * 100).toFixed(1) + '%';
  const domLabels = { extractive: '⛏ Extractive', enumeration: '📋 Enumeration', abstractive: '✍ Abstractive' };
  document.getElementById('tri-dominant').textContent =
    (domLabels[cl.dominant] || cl.dominant) +
    `  (ext ${cl.w_ext.toFixed(2)} / enum ${cl.w_enum.toFixed(2)} / abs ${cl.w_abs.toFixed(2)})`;
  document.getElementById('w-alpha').textContent = cl.alpha.toFixed(4);
  document.getElementById('w-beta').textContent  = cl.beta.toFixed(4);
  document.getElementById('w-gamma').textContent = cl.gamma.toFixed(4);

  // Selection method badge
  const mBadge = document.getElementById('method-badge');
  const mLabel = document.getElementById('method-label');
  const method = data.selection_method || 'unknown';
  mBadge.className = 'method-badge ' + (method === 'extractive' ? 'kneedle' : 'marginal');
  const mDisplay = { extractive: '⛏ Extractive', enumeration: '📋 Enumeration', abstractive: '✍ Abstractive' };
  mLabel.textContent = mDisplay[method] || method;

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
