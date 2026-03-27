# VLM Profiler Report

## Baseline Comparison (batch=1, res=224, GPU)

| Model                   |   Latency (ms) |   WER |   Energy (J/inf) |             FLOPs | Params   |
|:------------------------|---------------:|------:|-----------------:|------------------:|:---------|
| blip2-opt-2.7b          |         115.24 |  1.39 |            29.34 |   773000000000.00 | 3745M    |
| blip2-flan-t5-xl        |         135.75 |  1.21 |            31.21 |     7880000000.00 | 3942M    |
| instructblip-flan-t5-xl |         329.69 |  1.33 |            69.56 |     8050000000.00 | 4023M    |
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

## Quality (WER) vs Latency — by Dataset

![Quality (WER) vs Latency — by Dataset](quality_vs_latency_by_dataset.png)

## FLOPs Comparison

![FLOPs Comparison](flops_comparison.png)

## Parameter Distribution by Component

![Parameter Distribution by Component](param_breakdown.png)

## CPU vs GPU Latency

![CPU vs GPU Latency](cpu_vs_gpu.png)

## Known Limitations

### Model-specific incompatibilities

| Model | Issue | Affected experiments |
|-------|-------|---------------------|
| adept/fuyu-8b | FP16 causes dtype mismatch (Float vs Half) | 3 (fp16 x 3 datasets) |
| adept/fuyu-8b | Processor returns lists for batched inputs | 9 (batch>1 x 3 datasets) |
| blip2-flan-t5-xl | torch.compile fails (T5 architecture) | 3 (torch_compile x 3 datasets) |
| instructblip-flan-t5-xl | torch.compile fails (T5 architecture) | 3 (torch_compile x 3 datasets) |

### Other notes

- **Resolution scaling is flat** for BLIP2/InstructBLIP — they resize internally to fixed vision encoder resolution
- **FLOPs**: exact measurement (calflops) only for some models; T5-based use rough estimate (2 x params)
- **FP16 can be slower** on T5-based models due to internal dtype casting overhead
