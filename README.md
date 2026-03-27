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
| idefics2-8b | 8,403M | IDEFICS2 |

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
```

## Current status

**Done:**
- Full profiling pipeline (latency, FLOPs, energy, quality)
- 8 models tested across 3 datasets, ~360 experiments
- Report with 10 charts and per-chart analysis
- PDF report ready for submission

**Known issues:**
- fuyu-8b: FP16 broken (dtype mismatch), batching broken (processor returns lists)
- T5-based models: torch.compile fails, FP16 adds overhead instead of speedup

## Planned

- Add FlashAttention-2 experiments for compatible models
- Per-component latency breakdown (vision encoder vs LLM backbone vs cross-attention)
- Test additional models (moondream2, cogagent-vqa-hf) that were dropped due to compatibility issues
- Throughput analysis (tokens/second) alongside latency
- Quantization experiments (INT8, GPTQ)

## Hardware

2x NVIDIA RTX 5880 Ada (48 GB VRAM each), Python 3.13, PyTorch + HuggingFace Transformers.
