# OTTER — Open Training-free Text Encoder and Retrieval

OTTER is a training-free prompt compression pipeline that reduces long-context inputs before they reach an LLM reader, preserving task-relevant information while cutting token count. It requires no fine-tuning and runs fully locally. 

## How it works 

Each document is segmented into sentences, encoded into dense embeddings, and scored across three dimensions:

- **Anchor** — positional bias toward document boundaries
- **Flow** — neighbourhood coherence between adjacent sentences
- **Flash** — semantic similarity to the query

A `QueryClassifier` detects whether the query is extractive, enumerative, or abstractive and adjusts the scoring weights accordingly. The top-scoring sentences are selected and reassembled as the compressed output.

## Project layout

```
otter/
├── src/
│   ├── compress.py       # OTTERCompressor — full pipeline
│   ├── encoder.py        # SentenceEncoder (all-MiniLM-L6-v2)
│   ├── segmenter.py      # SentenceSegmenter (spaCy)
│   ├── classifier.py     # QueryClassifier
│   ├── planner.py        # QueryPlanner (scoring + selection)
│   └── device.py         # MPS / CUDA / CPU auto-detection
├── data/
│   ├── inspect.py        # dataset stats & summary CSV
│   └── dataset_summary.csv
├── results/              # benchmark outputs (git-ignored)
├── beaver/               # Beaver benchmarking model
├── loader.py             # Docling document loader (PDF, DOCX, PPTX, HTML)
├── mcp_server.py         # MCP server (otter_compress / otter_compare / otter_compress_document)
├── .mcp.json             # Claude Code MCP config
├── app.py                # Flask debugger UI
├── benchmark.py          # OTTER vs Beaver vs baseline evaluation
├── evaluate.py           # QwenEvaluator (Qwen2.5-3B-Instruct)
└── requirements.txt
```

## Setup

Requires **Python 3.10+**. Use conda:

```bash
conda create -n otter-env python=3.10 -y
conda activate otter-env
conda install pytorch=2.5.1 torchvision -c conda-forge -y
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## MCP Server

OTTER ships as a [Model Context Protocol](https://modelcontextprotocol.io) server, letting Claude Code call compression directly as a tool.

### Tools

| Tool | Description |
|---|---|
| `otter_compress` | Compress a text passage; set `compress=False` for an unmodified baseline with token counts |
| `otter_compare` | Returns original and compressed text side-by-side with token stats in one call |
| `otter_compress_document` | Fetches a URL or loads a file (PDF, DOCX, PPTX, HTML) via Docling, then compresses |

### Setup

```bash
pip install "mcp[cli]" tiktoken
```

Add `.mcp.json` to the repo root (or copy from the provided template):

```json
{
  "mcpServers": {
    "otter": {
      "type": "stdio",
      "command": "/path/to/your/env/bin/python",
      "args": ["/path/to/otter/mcp_server.py"],
      "env": {}
    }
  }
}
```

Then restart Claude Code — the three tools will appear automatically.

### Example

```
otter_compress(
    text="<long document>",
    query="What are the key findings?",
)
# → { text, original_tokens, compressed_tokens, reduction_pct, ... }
```

## Debugger UI

`app.py` is an interactive Flask app for testing and visualising the compression pipeline.

```bash
python app.py
# → http://localhost:5001
```

Features:
- Paste text or upload a file (PDF, DOCX, PPTX, HTML) — Docling extracts clean text automatically
- Per-sentence score visualisation (Anchor / Flow / Flash breakdown)
- Query classifier weights and dominant class display
- Compression ratio and token reduction stats

## Usage

**One-off compression:**
```python
from src.compress import compress
result = compress(document, query)
print(result["compressed"])
print(f"Kept {result['kept_sentences']}/{result['original_sentences']} sentences")
print(f"Token reduction: {result['token_reduction_pct']:.1f}%")
```

**Reusable compressor (models load once):**
```python
from src.compress import OTTERCompressor
compressor = OTTERCompressor()
for doc, query in pairs:
    result = compressor.compress(doc, query)
```

**Multi-document mode:**
```python
result = compressor.compress(document="", query=query,
                             documents=["doc1 text", "doc2 text"])
```

**With Docling (PDF / DOCX / URL):**
```python
from loader import DocumentLoader
from src.compress import OTTERCompressor

loader = DocumentLoader()
compressor = OTTERCompressor()

text = loader.load("paper.pdf")
result = compressor.compress(text, query)
```

## Download LongBench data

```python
from datasets import load_dataset

subsets = ['qasper', 'multifieldqa_en', 'qmsum', 'multi_news']
for subset in subsets:
    ds = load_dataset('THUDM/LongBench', subset, split='test', trust_remote_code=True)
    ds.to_json(f'data/longbench_{subset}.jsonl')
```

> **Note:** requires `datasets==2.21.0` — the 3.x/4.x series dropped support for the `LongBench.py` loading script.

## Benchmarking

```bash
# Run OTTER against all LongBench subsets
python benchmark.py --mode otter

# Run Beaver baseline
python benchmark.py --mode beaver

# Results written to results/
```
