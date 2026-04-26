# SafeAI Medical Brain v2 (capstone2)

This repository contains the complete offline-first medical brain pipeline for:

- Uganda Clinical Guidelines 2023
- WHO Malaria Guidelines (NCBI Bookshelf)

It includes:

- Python medical pipeline (`pipeline/`)
- FastAPI service (`api.py`)
- Optional local GGUF inference (`llama-cpp-python`)
- Windows WPF prototype app (`windows/Capstone2PipelineViewer/`)
- Technical architecture documentation (`ARCHITECTURE.md`)
- Full medical brain technical documentation (`MEDICAL_BRAIN_TECHNICAL_DOCUMENTATION.md`)

## Quick start

```powershell
cd c:\temp\capstone2
pip install -r requirements-api.txt
python -m uvicorn api:app --host 0.0.0.0 --port 8001
```

Open:

- API docs: `http://127.0.0.1:8001/docs`
- Health: `http://127.0.0.1:8001/health`

## Source data paths

- Local LLM GGUF: `C:\models\qwen2.5-3b-instruct-q4_k_m.gguf`
- Uganda PDF: `C:\temp\capstone\Uganda Clinical Guidelines 2023.pdf`
- WHO malaria PDF: `C:\temp\capstone\Bookshelf_NBK588130.pdf`

## Main workflows

- Build/load and ask via API:
  1. `POST /initialize`
  2. `POST /ask`
- One-shot CLI ask:
  - `python run_ask.py --preset uganda --query "When should I refer a child with fever?"`
- Desktop prototype:
  - Open `windows/Capstone2PipelineViewer/Capstone2PipelineViewer.sln`
  - Or run `windows/run_capstone2_viewer.ps1`

## Documentation map

- System architecture: `ARCHITECTURE.md`
- Complete technical documentation: `MEDICAL_BRAIN_TECHNICAL_DOCUMENTATION.md`
