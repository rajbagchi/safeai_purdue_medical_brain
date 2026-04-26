# Medical Pipeline Package

Modular production pipeline for WHO Malaria Guidelines: multi-pass extraction, validation, chunking, and two-brain Q&A.

## Source documents (config presets)

`ExtractionConfig` accepts any `pdf_path` string (Windows absolute paths, spaces, etc.). For the two validated PDFs on disk:

| Preset | Default PDF | Output dir (default) |
|--------|-------------|----------------------|
| `extraction_config_who_malaria_nih()` | `C:\temp\capstone\Bookshelf_NBK588130.pdf` | `C:\temp\capstone\medical_kb_who_malaria` |
| `extraction_config_uganda_clinical_2023()` | `C:\temp\capstone\Uganda Clinical Guidelines 2023.pdf` | `C:\temp\capstone\medical_kb_uganda_clinical_2023` |

Each preset sets `document_title` and `critical_content_terms` so Stage 4 medical-term checks are appropriate (malaria-specific vs. broad clinical). Paths are normalized in `ExtractionConfig.__post_init__` via `pathlib.Path.expanduser().resolve()`.

CLI:

```bash
python run_pipeline.py --preset who-malaria
python run_pipeline.py --preset uganda
python run_pipeline.py --pdf "D:\other\guide.pdf" --output-dir ./my_kb
```

## HTTP API (FastAPI)

Repo root `api.py` exposes `MedicalQASystem` over HTTP.

```bash
pip install -r requirements-api.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

| Method | Path | Purpose |
|--------|------|--------|
| `GET` | `/health` | Status + whether a pipeline is loaded |
| `GET` | `/metadata` | Preset, paths, chunk count, extraction summary (after init) |
| `POST` | `/initialize` | Body: `preset`, optional `pdf_path` / `output_dir`, `reuse_existing_kb` |
| `POST` | `/ask` | Body: `query`, optional `full_response`, optional `use_local_llm` (local GGUF simplifier) |

Presets: `who-malaria` (aliases `who-malaria-nih`, `malaria`), `uganda` (aliases `uganda-clinical-2023`).

**Windows PowerShell:** `curl` is an alias for `Invoke-WebRequest`. For **`curl.exe`**, PowerShell often **rewrites arguments** so `-d $json` still arrives at the server **without inner double quotes** (`422` / “Expecting property name enclosed in double quotes”) even when `$json` looks correct in the shell.

Use one of these instead:

```powershell
# Best on Windows: no curl quoting issues
Invoke-RestMethod http://localhost:8000/health

Invoke-RestMethod -Uri http://localhost:8000/initialize -Method Post `
  -ContentType application/json `
  -Body (@{ preset = "who-malaria-nih" } | ConvertTo-Json -Compress)

Invoke-RestMethod -Uri http://localhost:8000/ask -Method Post `
  -ContentType application/json `
  -Body (@{ query = "Child has fever and cannot drink. What should I do?" } | ConvertTo-Json -Compress)
```

```powershell
# curl.exe: read body from a file (repo includes samples under scripts/)
curl.exe http://localhost:8000/health

curl.exe -X POST http://localhost:8000/initialize `
  -H "Content-Type: application/json" `
  --data-binary "@scripts\api_sample_init.json"

curl.exe -X POST http://localhost:8000/ask `
  -H "Content-Type: application/json" `
  --data-binary "@scripts\api_sample_ask.json"
```

```powershell
# curl.exe: stop parsing so the rest of the line is passed literally to curl (no variables)
curl.exe --% -X POST http://localhost:8000/initialize -H "Content-Type: application/json" -d {"preset":"who-malaria-nih"}
```

On **Git Bash** or **Linux/macOS**, plain `curl -d '{"preset":"who-malaria-nih"}'` is fine; you can also use `--data-binary @scripts/api_sample_init.json` from the repo root.

Markdown reports (stages + 25 searches):

```bash
python scripts/who_malaria_pipeline_report.py --preset who-malaria
python scripts/who_malaria_pipeline_report.py --preset uganda --reuse-kb
```

## Response layer (VHT output)

After **indexing → guardrail**, the **output layer** formats community-facing text:

| Module | Role |
|--------|------|
| `retrieval.py` | `infer_query_intent()`, BM25 pool + heading-quality re-rank (`retrieve_top_chunk_indices`) |
| `response.py` | `ResponseOrchestrator`, `VHTResponseFormatter`, `ResponseContent`, `infer_triage_from_query` |
| `local_simplifier.py` | Optional **Qwen2.5-3B (GGUF)** rewrite: `LocalSimplifierLLM` + `llama-cpp-python` |
| `orchestrator.py` | `MedicalQASystem.answer_with_response()` — BM25 + guardrail + structured VHT + referral note |

Hospital-referral-style questions map to **YELLOW** triage and referral-oriented VHT steps (not generic “manage at home”). `answer_with_response` includes **`query_intent`**: `referral_hospital` \| `dosing` \| `general`.

### Local VHT simplifier (Qwen2.5 3B Instruct GGUF)

After `ResponseOrchestrator.create()` builds `ResponseContent`, an optional **local** model may rewrite **only** `vht_response` for readability. **Retrieval and the first guardrail pass are unchanged.** The LLM output must include the same section headers the guardrail expects; **`MedicalGuardrailBrain.validate_response()`** runs again on the LLM text. If the model fails, times out, or validation fails, the rule-based **`VHTResponseFormatter`** output is kept.

Install: `pip install -r requirements-local-llm.txt` (see file for Windows CPU wheel hint). Then:

| Env | Meaning |
|-----|--------|
| `SAFEAI_USE_LOCAL_LLM=1` | Enable when `SAFEAI_LLM_GGUF` points to a file |
| `SAFEAI_LLM_GGUF` | Full path to `.gguf` (e.g. Qwen2.5-3B-Instruct Q4_K_M) |
| `SAFEAI_LLM_N_THREADS` | CPU threads (default ~ half of cores) |
| `SAFEAI_LLM_N_CTX` | Context length (default 4096) |
| `SAFEAI_LLM_N_GPU_LAYERS` | Set `>0` only if you have GPU support in your build |

API: `POST /ask` with `"use_local_llm": true` forces the simplifier when the GGUF path exists; `false` disables; omit to follow env only. Response fields: **`local_llm_used`**, **`local_llm_skipped_reason`** (if not used).

`MedicalSource` and `medical_source_for_config()` in `config.py` label citations (WHO malaria vs Uganda CG).

```python
from pipeline import MedicalQASystem, extraction_config_who_malaria_nih

qa = MedicalQASystem(config=extraction_config_who_malaria_nih())
qa.initialize()
out = qa.answer_with_response(
    "What is ACT dosing for children?",
    use_local_llm=True,  # optional; requires SAFEAI_LLM_GGUF + llama-cpp-python
)
print(out["vht_response"])
print(out["referral_note"])
print(out["quick_summary"])
```

**Validate 25 test queries × both PDFs** (requires built KBs under default output dirs):

```bash
python scripts/response_layer_validation_report.py
```

Writes `reports/response_layer_validation.md` (and a timestamped copy): summary table plus **full query text** and **complete** `vht_response`, `referral_note`, `quick_summary`, and BM25 evidence bundle per query.

## Structure

| Module | Purpose |
|--------|---------|
| `config.py` | `ExtractionConfig`, `ValidationReport`, `TriageLevel`, `DangerSign` |
| `extractor.py` | `MultiPassExtractor` — PDF analysis, text/table/OCR extraction, cross-validation |
| `validator.py` | `ExtractionValidator` — structure, tables, cross-consistency, medical content, human-review flags |
| `chunker.py` | `SmartChunker` — semantic chunks by headings, BM25 search index |
| `retrieval.py` | Query intent + quality-weighted BM25 top-k |
| `local_simplifier.py` | Optional GGUF simplifier (`llama-cpp-python`) |
| `response.py` | VHT formatting, triage, `ResponseOrchestrator` |
| `guardrail.py` | `MedicalGuardrailBrain` — triage, dangerous advice, citations |
| `orchestrator.py` | `MedicalQASystem` — runs pipeline, saves/loads KB, `answer()` |
| `cli.py` | Interactive Q&A entry point |
| `__main__.py` | Enables `python -m pipeline` |

## Usage

From project root:

```bash
# Run interactive Q&A (builds or loads knowledge base)
python run_pipeline.py

# Or
python -m pipeline
```

In code:

```python
from pipeline import MedicalQASystem, ExtractionConfig

qa = MedicalQASystem("path/to/guidelines.pdf", output_dir="./medical_knowledge_base")
qa.initialize()
result = qa.answer("What is the dose for severe malaria in children?")
```

## Dependencies

Install from repo root:

```bash
pip install -r requirements-pipeline.txt
```

Includes: **PyMuPDF**, **numpy**, **pandas**, **rank-bm25**, **rapidfuzz**, **tabulate** (for `DataFrame.to_markdown` on tables), **pdfplumber** (cross-validation vs. Pass 1 text). Optional: **camelot** for borderless tables.

Extractor behavior:

- **Full-document table scan** (`ExtractionConfig.full_document_table_scan`, default `true`) finds all pages with PyMuPDF tables (not only the first 20 sampled in Pass 0).
- **Embedded images** saved under `{output_dir}/images/` as PNG plus `image_inventory.json` when `enable_image_extraction` is true.
