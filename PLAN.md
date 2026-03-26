# VLM Profiler — Implementation Plan

## Requirements Restatement

**Goal:** Profile Vision-Language Models (VLMs) to understand how they scale, identify bottlenecks, and suggest optimizations for latency and energy consumption.

**What we measure per model:**
- Latency (ms) — end-to-end and per-component (vision encoder, LLM backbone, cross-attention)
- FLOPs — theoretical and measured
- Energy consumption (W, J) — via `nvidia-smi` power draw
- Quality — WER (Word Error Rate) or task-specific accuracy

**Scaling axes:**
| Axis | Values |
|------|--------|
| Image resolution | 224, 336, 448, 512 |
| Prompt length | 10, 50, 100, 200 tokens |
| Device | CPU vs GPU |
| Optimizations | FP32 baseline, FP16, torch.compile, FlashAttention-2 |
| Batch size | 1, 2, 4, 8 |

**Models (10):**
1. `llava-hf/llava-1.5-400m-fp16` (tiny)
2. `llava-hf/llava-1.5-13b-hf` (large)
3. `Salesforce/blip2-opt-2.7b`
4. `Salesforce/instructblip-vicuna-7b`
5. `Salesforce/instructblip-flan-t5-xl`
6. `Salesforce/blip2-flan-t5-xl`
7. `adept/fuyu-8b`
8. `THUDM/cogagent-vqa-hf`
9. `vikhyatk/moondream2`
10. `HuggingFaceM4/idefics2-8b`

**Datasets (300 samples each):**
1. ScienceQA (mini-val)
2. TextVQA (val)
3. COCO Caption Val (subset)

**Deliverable:** Report with measurements, charts, and recommendations on which models are optimal for speed/quality/energy, and which model components need the most optimization.

---

## Environment

- **GPUs:** 2x NVIDIA RTX 5880 Ada (48GB VRAM each)
- **Python:** 3.13 (conda)
- **CUDA:** 13.0

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Some models don't support variable resolution | HIGH | Check processor config, skip unsupported resolutions gracefully |
| `cogagent-vqa-hf` may need special dependencies | MEDIUM | Isolate in try/except, document requirements |
| FP16 + CPU is meaningless | LOW | Skip CPU+FP16 combination |
| OOM on batch_size=8 for 13B models | HIGH | Catch OOM, reduce batch, log as "OOM" |
| `torch.compile` may fail on some architectures | MEDIUM | Wrap in try/except, fall back to eager |
| Energy measurement via nvidia-smi is approximate | LOW | Document precision limits, use consistent measurement protocol |
| FlashAttention-2 not available for all model architectures | MEDIUM | Check compatibility, skip gracefully |

---

## Project Structure

```
vlm-profiler/
├── PLAN.md                   # This file
├── README.md
├── requirements.txt
├── configs/
│   └── experiment_config.yaml   # All models, datasets, scaling axes
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py            # Dataset loading (ScienceQA, TextVQA, COCO)
│   │   └── preprocessing.py     # Image resize, prompt generation
│   ├── models/
│   │   ├── __init__.py
│   │   └── registry.py          # Model loading with optimization variants
│   ├── profiling/
│   │   ├── __init__.py
│   │   ├── latency.py           # Latency measurement (warmup + timing)
│   │   ├── flops.py             # FLOPs counting (fvcore / calflops)
│   │   ├── energy.py            # GPU power monitoring via nvidia-smi
│   │   └── quality.py           # WER / accuracy metrics
│   ├── runner.py                # Main experiment loop
│   └── utils.py                 # GPU utils, logging, OOM handling
├── scripts/
│   ├── run_experiments.py       # Entry point
│   └── generate_report.py       # Aggregate results → charts + markdown report
├── results/                     # Raw JSON results per experiment
├── reports/                     # Generated charts and final report
└── tests/
    ├── test_loader.py
    ├── test_profiling.py
    └── test_runner.py
```

---

## Implementation Phases

### Phase 0: Environment Setup
- [ ] Create conda environment with Python 3.11 (better library compat than 3.13)
- [ ] Install core deps: `torch`, `transformers`, `accelerate`, `Pillow`, `datasets`
- [ ] Install profiling deps: `fvcore` or `calflops`, `jiwer` (for WER)
- [ ] Install optional: `flash-attn` (if compatible)
- [ ] Verify GPU access with a simple torch test
- [ ] Create `requirements.txt`

### Phase 1: Data Pipeline
- [ ] Write `src/data/loader.py` — download and cache 300 samples from each dataset
  - ScienceQA: filter for image-based questions, extract image + question + answer
  - TextVQA: load val split, sample 300
  - COCO Captions: load val split, sample 300 image-caption pairs
- [ ] Write `src/data/preprocessing.py` — resize images to target resolutions
- [ ] Write tests for data loading

### Phase 2: Model Registry
- [ ] Write `src/models/registry.py` — unified interface to load any of the 10 models
  - `load_model(name, device, dtype, use_compile, use_flash_attn) → (model, processor)`
  - Handle model-specific quirks (cogagent needs trust_remote_code, etc.)
- [ ] Implement optimization variants: FP16 casting, torch.compile wrapping, FA2 toggle
- [ ] Write tests: verify each model loads and produces output on a single sample

### Phase 3: Profiling Modules
- [ ] `src/profiling/latency.py`
  - Warmup runs (3-5), then timed runs (N=10+)
  - Measure total latency and per-component if possible (hook into model forward)
  - Return mean, std, p50, p95, p99
- [ ] `src/profiling/flops.py`
  - Use `fvcore.nn.FlopCountAnalysis` or `calflops` to estimate FLOPs
  - Handle models that don't support standard flop counting gracefully
- [ ] `src/profiling/energy.py`
  - Background thread polling `nvidia-smi --query-gpu=power.draw --format=csv` at 100ms intervals
  - Compute average power (W) during inference → multiply by time → energy (J)
- [ ] `src/profiling/quality.py`
  - For VQA tasks: exact-match accuracy
  - For captioning: WER via `jiwer`, optionally BLEU/CIDEr
- [ ] Write tests for each profiling module

### Phase 4: Experiment Runner
- [ ] Write `configs/experiment_config.yaml` — full experiment matrix
- [ ] Write `src/runner.py`
  - Load config → iterate over (model, dataset, resolution, prompt_len, device, optimization, batch_size)
  - Skip invalid combos (CPU+FP16, CPU+FlashAttn, etc.)
  - Catch OOM → log and continue
  - Save results as JSON per experiment run
- [ ] Write `scripts/run_experiments.py` — CLI entry point with filtering options
  - `--models`, `--datasets`, `--resolutions`, `--device`, etc.
  - Resume support (skip already-completed experiments)

### Phase 5: Report Generation
- [ ] Write `scripts/generate_report.py`
  - Load all JSON results
  - Generate charts (matplotlib/plotly):
    - Latency vs resolution per model
    - Latency vs batch size per model
    - Latency vs prompt length per model
    - FLOPs comparison across models
    - Energy consumption comparison
    - Quality vs latency scatter (Pareto frontier)
    - Speedup from optimizations (FP16, compile, FA2) per model
    - CPU vs GPU comparison
  - Generate markdown report with embedded charts
  - Identify top-3 models per metric, overall Pareto-optimal models
  - Identify bottleneck components per model

### Phase 6: Final Analysis
- [ ] Run full experiment suite
- [ ] Generate report
- [ ] Write conclusions:
  - Which models are Pareto-optimal (quality/speed/energy)?
  - Which model components are the bottleneck?
  - Which optimizations give the best speedup?
  - Recommendations for production deployment

---

## Execution Order & Dependencies

```
Phase 0 (env setup)
    ↓
Phase 1 (data) ──────┐
    ↓                 │
Phase 2 (models) ─────┤
    ↓                 │
Phase 3 (profiling) ──┘
    ↓
Phase 4 (runner) — depends on 1, 2, 3
    ↓
Phase 5 (report gen) — depends on 4 results
    ↓
Phase 6 (analysis) — depends on 5
```

Phases 1, 2, 3 can be developed in parallel since they are independent modules.

---

## Key Design Decisions

1. **YAML config over CLI flags** — the experiment matrix is large; a config file is easier to manage and reproduce
2. **JSON results per experiment** — allows resume and incremental analysis without re-running everything
3. **Separate report generation** — decouple measurement from visualization; can re-generate reports without re-running
4. **Graceful degradation** — every measurement wraps in try/except; unsupported combos are logged, not crashed
5. **Python 3.11 conda env** — better compatibility with torch, transformers, flash-attn than 3.13

---

**WAITING FOR CONFIRMATION**: Proceed with this plan? (yes / no / modify)
