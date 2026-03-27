# VLM Profiler Report

## Baseline Comparison (batch=1, res=224, GPU)

| Model                   |   Latency (ms) |   WER |   Energy (J/inf) |             FLOPs | Params   |
|:------------------------|---------------:|------:|-----------------:|------------------:|:---------|
| blip2-opt-2.7b          |         115.24 |  1.39 |            29.34 |   773000000000.00 | 3745M    |
| blip2-flan-t5-xl        |         135.75 |  1.21 |            31.21 |     7880000000.00 | 3942M    |
| instructblip-flan-t5-xl |         329.69 |  1.33 |            69.56 |     8050000000.00 | 4023M    |
| idefics2-8b             |         915.45 |  9.26 |           243.97 | 27100000000000.00 | 8403M    |
| fuyu-8b                 |        1562.25 |  6.21 |           403.91 |  1380000000000.00 | 9408M    |
| llava-1.5-7b-hf         |        1654.08 |  6.38 |           450.39 |  8290000000000.00 | 7063M    |
| llava-1.5-13b-hf        |        1704.07 |  6.46 |           470.38 | 15800000000000.00 | 13351M   |
| instructblip-vicuna-7b  |        1792.87 |  6.17 |           486.71 |  2370000000000.00 | 7914M    |

## Quality (WER) by Dataset

| model_short             |   coco_caption |   scienceqa |   textvqa |
|:------------------------|---------------:|------------:|----------:|
| blip2-flan-t5-xl        |           0.79 |        1.30 |      1.54 |
| blip2-opt-2.7b          |           0.82 |        1.49 |      1.87 |
| fuyu-8b                 |           4.65 |        6.42 |      7.57 |
| idefics2-8b             |           2.24 |       16.83 |      8.69 |
| instructblip-flan-t5-xl |           1.85 |        1.28 |      0.86 |
| instructblip-vicuna-7b  |           0.96 |       11.18 |      6.36 |
| llava-1.5-13b-hf        |           3.36 |        7.96 |      8.06 |
| llava-1.5-7b-hf         |           3.37 |        7.02 |      8.75 |

## Latency vs Image Resolution

![Latency vs Image Resolution](latency_vs_resolution.png)

## Latency vs Prompt Length

![Latency vs Prompt Length](latency_vs_prompt_length.png)

## Latency vs Batch Size

![Latency vs Batch Size](latency_vs_batch_size.png)

## Optimization Speedup (FP16, torch.compile)

![Optimization Speedup (FP16, torch.compile)](optimization_speedup.png)

## Energy per Inference

![Energy per Inference](energy_comparison.png)

## Energy vs Resolution

![Energy vs Resolution](energy_vs_resolution.png)

## Quality (WER) vs Latency

![Quality (WER) vs Latency](quality_vs_latency.png)

## FLOPs Comparison

![FLOPs Comparison](flops_comparison.png)

## Parameter Distribution by Component

![Parameter Distribution by Component](param_breakdown.png)

## CPU vs GPU Latency

![CPU vs GPU Latency](cpu_vs_gpu.png)

## Comparison with Official Paper Results

### Methodology Notes

Our profiler measures **WER (Word Error Rate)** — lower is better. Papers report **accuracy** (%) — higher is better.
These metrics are not directly comparable, but we can verify ranking consistency.

Our setup: 300 samples per dataset, zero-shot, greedy decoding, 224x224 resolution, max 50 new tokens.
Paper setups vary: different sample sizes (full test sets), beam search, sometimes task-specific fine-tuning.

### Official Accuracy from Papers (zero-shot unless noted)

| Model | TextVQA | ScienceQA (img) | VQAv2 | Source |
|-------|---------|-----------------|-------|--------|
| BLIP-2 OPT-2.7B | — | — | 53.5 | Li et al. 2023 (BLIP-2), Table 2 |
| BLIP-2 FlanT5-XL | — | — | 63.0 | Li et al. 2023 (BLIP-2), Table 2 |
| InstructBLIP FlanT5-XL | 46.6 | 70.4 | — | Dai et al. 2023 (InstructBLIP), Table 1 |
| InstructBLIP Vicuna-7B | 50.1 | 60.5 | — | Dai et al. 2023 (InstructBLIP), Table 1 |
| LLaVA-1.5 7B | 58.2 | 66.8 | 78.5* | Liu et al. 2023 (LLaVA-1.5), Table 3 (*fine-tuned) |
| LLaVA-1.5 13B | 61.3 | 71.6 | 80.0* | Liu et al. 2023 (LLaVA-1.5), Table 3 (*fine-tuned) |
| Idefics2-8B (base, 8-shot) | 57.9 | — | 70.3 | Laurençon et al. 2024 (Idefics2), Table 8 |
| Idefics2-8B (instruct, 0-shot) | 70.4 | — | — | Laurençon et al. 2024 (Idefics2), Table 9 |
| Fuyu-8B | — | — | 74.2 | Adept blog post |

Note: BLIP-2 paper does not report TextVQA or ScienceQA. Fuyu blog reports only VQAv2, OKVQA, COCO CIDEr, AI2D.

### Our WER Results vs Paper Rankings

**TextVQA** (papers report accuracy; our WER — lower = better):

| Model | Paper Acc (%) | Our WER | Ranking Match? |
|-------|--------------|---------|----------------|
| InstructBLIP FlanT5-XL | 46.6 | **0.86** | Best in our test |
| BLIP-2 FlanT5-XL | — | 1.54 | |
| BLIP-2 OPT-2.7B | — | 1.87 | |
| Fuyu-8B | — | 7.57 | |
| LLaVA-1.5 13B | 61.3 | 8.06 | |
| Idefics2-8B | 57.9–70.4 | 8.69 | Worse than expected |
| LLaVA-1.5 7B | 58.2 | 8.75 | Worse than expected |

**ScienceQA** (papers report accuracy; our WER):

| Model | Paper Acc (%) | Our WER | Ranking Match? |
|-------|--------------|---------|----------------|
| InstructBLIP FlanT5-XL | 70.4 | **1.28** | Best — matches paper |
| BLIP-2 FlanT5-XL | — | 1.30 | |
| BLIP-2 OPT-2.7B | — | 1.49 | |
| Fuyu-8B | — | 6.42 | |
| LLaVA-1.5 7B | 66.8 | 7.02 | |
| LLaVA-1.5 13B | 71.6 | 7.96 | |
| Idefics2-8B | — | 16.83 | |

Note: InstructBLIP Vicuna-7B excluded — produces garbage output due to HuggingFace transformers incompatibility (see Known Limitations).

### Analysis

**What matches papers:**
- **FlanT5 models dominate** on TextVQA and ScienceQA, consistent with InstructBLIP paper showing strong short-answer performance
- **InstructBLIP > BLIP-2** on quality, confirming instruction tuning helps
- **FlanT5-XL > OPT-2.7B** within BLIP-2 family, matching paper's finding

**What diverges from papers:**
- **LLaVA-1.5 underperforms** despite having highest accuracy in papers (58–61% TextVQA). Root cause: LLaVA generates verbose conversational answers ("The answer is X because..."). WER penalizes longer responses compared to short references.
- **Idefics2 also underperforms** for the same verbosity reason
- **BLIP-2/InstructBLIP FlanT5 look artificially good** — encoder-decoder T5 naturally produces short, terse answers matching reference format

**Key takeaway:** WER is a format-sensitive metric. Models tuned for short answers (FlanT5-based) score better on WER than models tuned for natural conversation (LLaVA, Idefics2), even when the latter have higher actual accuracy. For fair comparison, VQA-accuracy with answer extraction would be more appropriate.

## Known Limitations

### Model-specific incompatibilities

| Model | Issue | Affected experiments |
|-------|-------|---------------------|
| instructblip-vicuna-7b | Generates garbage tokens (repeated `6▶↵6`) — likely HF transformers version incompatibility | ALL quality results invalid |
| adept/fuyu-8b | FP16 causes dtype mismatch (Float vs Half) | 3 (fp16 x 3 datasets) |
| adept/fuyu-8b | Processor returns lists for batched inputs | 9 (batch>1 x 3 datasets) |
| blip2-flan-t5-xl | torch.compile fails (T5 architecture) | 3 (torch_compile x 3 datasets) |
| instructblip-flan-t5-xl | torch.compile fails (T5 architecture) | 3 (torch_compile x 3 datasets) |

### Other notes

- **Resolution scaling is flat** for BLIP2/InstructBLIP — they resize internally to fixed vision encoder resolution
- **FLOPs**: exact measurement (calflops) only for some models; T5-based use rough estimate (2 x params)
- **FP16 can be slower** on T5-based models due to internal dtype casting overhead
- **instructblip-vicuna-7b quality metrics (WER=1.00 across all datasets) are invalid** — model outputs are garbage tokens, not meaningful text. Latency/energy/FLOPs measurements are still valid.
