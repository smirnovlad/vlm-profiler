# VLM Profiler

Benchmarking toolkit for Vision-Language Models. Measures latency, FLOPs, GPU energy consumption, and output quality (WER) across different configurations.

## What this does

Runs a matrix of experiments on VLMs, varying one parameter at a time:

- **Image resolution**: 224, 336, 448, 512
- **Prompt length**: 10, 50, 100, 200 tokens
- **Batch size**: 1, 2, 4, 8
- **Optimizations**: FP32, FP16, torch.compile
- **Device**: CUDA, CPU

Each experiment records latency (mean/std/percentiles), energy draw via `nvidia-smi`, FLOPs (calflops), and WER against reference answers. Results are saved as JSON files per run, then aggregated into charts and a markdown report.

## Models

| Model | Params | Backbone |
|-------|--------|----------|
| blip2-opt-2.7b | 3,745M | BLIP-2 + OPT |
| blip2-flan-t5-xl | 3,942M | BLIP-2 + Flan-T5 |
| instructblip-flan-t5-xl | 4,023M | InstructBLIP + Flan-T5 |
| instructblip-vicuna-7b | 7,914M | InstructBLIP + Vicuna |
| llava-1.5-7b-hf | 7,063M | LLaVA 1.5 |
| llava-1.5-13b-hf | 13,351M | LLaVA 1.5 |
| fuyu-8b | 9,408M | Fuyu |

`idefics2-8b` was in the original plan but dropped: every generation call crashes with a pixel-values shape mismatch against transformers 5.x. Tracked as a known compatibility issue, not included in results.

## Datasets

300 samples each from ScienceQA (val), TextVQA (val), and COCO Captions (val).

## Project structure

```
vlm-profiler/
├── configs/
│   └── experiment_config.yaml   # full experiment matrix
├── src/
│   ├── data/
│   │   ├── loader.py            # dataset loading
│   │   └── preprocessing.py     # image resize, prompt templates
│   ├── models/
│   │   └── registry.py          # model loading with optimizations
│   ├── profiling/
│   │   ├── latency.py           # warmup + timed inference
│   │   ├── flops.py             # FLOPs via calflops
│   │   ├── energy.py            # GPU power monitoring
│   │   └── quality.py           # WER computation
│   ├── runner.py                # experiment loop
│   └── utils.py                 # GPU utils, OOM handling
├── scripts/
│   ├── run_experiments.py       # CLI entry point
│   └── generate_report.py       # results -> charts + markdown
├── results/                     # generated charts and REPORT.md
├── latex_report/                # PDF report (tectonic)
└── outputs/                     # raw JSON results per experiment
```

## Setup

```bash
conda create -n vlm-profiler python=3.11
conda activate vlm-profiler
pip install -r requirements.txt
```

## Usage

Run experiments:
```bash
python scripts/run_experiments.py --config configs/experiment_config.yaml --device cuda
```

Generate report from results:
```bash
python scripts/generate_report.py --outputs-root outputs/2026-03-27 --report-dir results
```

Compile PDF report:
```bash
cd latex_report && tectonic report.tex
mv report.pdf ../report.pdf
```

## Status

- Full profiling pipeline (latency, FLOPs, energy, quality) on 7 models × 3 datasets, ~330 successful experiments.
- Per-component latency breakdown (vision encoder / projector / LLM decoder) via forward hooks + CUDA events.
- Prefill vs. per-token decode split with crossover analysis.
- FlashAttention-2 vs. eager attention A/B (via SDPA dispatch on Ada).
- Report at [`report.pdf`](report.pdf), markdown summary at [`results/REPORT.md`](results/REPORT.md).

**Known incompatibilities (documented in report):**
- `idefics2-8b`: pixel-values shape mismatch on transformers 5.x — dropped from results.
- `fuyu-8b`: FP16 dtype mismatch; processor doesn't support batched inputs.
- T5-based models: `torch.compile` fails; FP16 adds dtype-cast overhead instead of speedup.
- `flash-attn` pip package requires nvcc ≥ 11.7; system has 11.5 — used SDPA-FA2 dispatch instead.

## Hardware

2x NVIDIA RTX 5880 Ada (48 GB VRAM each), Python 3.13, PyTorch + HuggingFace Transformers.
