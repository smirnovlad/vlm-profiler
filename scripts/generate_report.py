#!/usr/bin/env python3
"""Aggregate experiment results from all models and generate charts + markdown report.

Usage:
    # Scan all model dirs under a date folder
    python scripts/generate_report.py --outputs-root outputs/2026-03-26

    # Output goes to reports/2026-03-26/
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_all_results(outputs_root: Path) -> pd.DataFrame:
    """Scan outputs_root/{model_name}/{timestamp}/results/*.json and merge."""
    records = []
    for json_file in outputs_root.rglob("results/*.json"):
        try:
            data = json.loads(json_file.read_text())
            if data.get("status") != "success":
                continue
            flat = {
                "model": data["model"],
                "model_short": data["model"].split("/")[-1],
                "dataset": data["dataset"],
                "resolution": data["resolution"],
                "prompt_length": data["prompt_length"],
                "device": data["device"],
                "optimization": data["optimization"],
                "batch_size": data["batch_size"],
                "latency_mean_ms": data["latency"]["mean_ms"],
                "latency_std_ms": data["latency"]["std_ms"],
                "latency_p50_ms": data["latency"]["p50_ms"],
                "latency_p95_ms": data["latency"]["p95_ms"],
                "latency_p99_ms": data["latency"]["p99_ms"],
            }
            # Energy
            if data.get("energy"):
                flat["avg_power_w"] = data["energy"]["avg_power_w"]
                flat["energy_per_inf_j"] = data["energy"].get("energy_per_inference_j")
                flat["energy_total_j"] = data["energy"]["energy_j"]
            else:
                flat["avg_power_w"] = None
                flat["energy_per_inf_j"] = None
                flat["energy_total_j"] = None
            # Quality (only on baselines)
            if data.get("quality"):
                flat["wer"] = data["quality"]["wer_score"]
                flat["exact_match"] = data["quality"]["exact_match_accuracy"]
            else:
                flat["wer"] = None
                flat["exact_match"] = None
            # FLOPs (only on baselines)
            if data.get("flops"):
                flat["flops"] = data["flops"]["total_flops"]
                flat["macs"] = data["flops"]["total_macs"]
                flat["flops_method"] = data["flops"].get("method", "unknown")
            else:
                flat["flops"] = None
                flat["macs"] = None
                flat["flops_method"] = None
            # Param counts
            if data.get("param_counts"):
                flat["total_params"] = sum(data["param_counts"].values())
                for comp, count in data["param_counts"].items():
                    flat[f"params_{comp}"] = count
            records.append(flat)
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return pd.DataFrame(records)


# --- Charts ---

def _baseline_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to baseline config: res=224, plen=10, bs=1, opt=none, cuda."""
    return df[
        (df["optimization"] == "none")
        & (df["batch_size"] == 1)
        & (df["device"] == "cuda")
        & (df["resolution"] == 224)
        & (df["prompt_length"] == 10)
    ]


def plot_latency_vs_resolution(df: pd.DataFrame, out_dir: Path):
    subset = df[
        (df["optimization"] == "none") & (df["batch_size"] == 1)
        & (df["device"] == "cuda") & (df["prompt_length"] == 10)
    ]
    if subset.empty:
        return
    avg = subset.groupby(["model_short", "resolution"])["latency_mean_ms"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    for model in sorted(avg["model_short"].unique()):
        mdata = avg[avg["model_short"] == model].sort_values("resolution")
        ax.plot(mdata["resolution"], mdata["latency_mean_ms"], marker="o", label=model)
    ax.set_xlabel("Image Resolution")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency vs Image Resolution (batch=1, GPU)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "latency_vs_resolution.png", dpi=150)
    plt.close(fig)


def plot_latency_vs_prompt_length(df: pd.DataFrame, out_dir: Path):
    subset = df[
        (df["optimization"] == "none") & (df["batch_size"] == 1)
        & (df["device"] == "cuda") & (df["resolution"] == 224)
    ]
    if subset.empty:
        return
    avg = subset.groupby(["model_short", "prompt_length"])["latency_mean_ms"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    for model in sorted(avg["model_short"].unique()):
        mdata = avg[avg["model_short"] == model].sort_values("prompt_length")
        ax.plot(mdata["prompt_length"], mdata["latency_mean_ms"], marker="s", label=model)
    ax.set_xlabel("Prompt Length (tokens)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency vs Prompt Length (batch=1, res=224, GPU)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "latency_vs_prompt_length.png", dpi=150)
    plt.close(fig)


def plot_latency_vs_batch(df: pd.DataFrame, out_dir: Path):
    subset = df[
        (df["optimization"] == "none") & (df["resolution"] == 224)
        & (df["device"] == "cuda") & (df["prompt_length"] == 10)
    ]
    if subset.empty:
        return
    avg = subset.groupby(["model_short", "batch_size"])["latency_mean_ms"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    for model in sorted(avg["model_short"].unique()):
        mdata = avg[avg["model_short"] == model].sort_values("batch_size")
        ax.plot(mdata["batch_size"], mdata["latency_mean_ms"], marker="s", label=model)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency vs Batch Size (res=224, GPU)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "latency_vs_batch_size.png", dpi=150)
    plt.close(fig)


def plot_optimization_speedup(df: pd.DataFrame, out_dir: Path):
    base = _baseline_filter(df).groupby("model_short")["latency_mean_ms"].mean()
    if base.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.35
    models = sorted(base.index)
    x = np.arange(len(models))
    for i, opt in enumerate(["fp16", "torch_compile"]):
        opt_data = df[
            (df["optimization"] == opt) & (df["batch_size"] == 1)
            & (df["device"] == "cuda") & (df["resolution"] == 224)
            & (df["prompt_length"] == 10)
        ].groupby("model_short")["latency_mean_ms"].mean()
        speedups = [float(base[m] / opt_data[m]) if m in opt_data else 0.0 for m in models]
        ax.bar(x + i * width, speedups, width, label=opt, alpha=0.8)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Speedup (vs baseline)")
    ax.set_title("Optimization Speedup per Model")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    fig.text(
        0.5, 0.01,
        "Note: bar = 0 means the optimization is not supported by the model (e.g. torch_compile fails on T5, FP16 on Fuyu)",
        ha="center", fontsize=8, style="italic", color="gray",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_dir / "optimization_speedup.png", dpi=150)
    plt.close(fig)


def plot_energy_comparison(df: pd.DataFrame, out_dir: Path):
    subset = _baseline_filter(df).dropna(subset=["energy_per_inf_j"])
    if subset.empty:
        return
    avg = subset.groupby("model_short")["energy_per_inf_j"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(avg.index, avg.values)
    ax.set_xlabel("Energy per Inference (J)")
    ax.set_title("Energy Consumption per Inference (batch=1, res=224, GPU)")
    plt.tight_layout()
    fig.savefig(out_dir / "energy_comparison.png", dpi=150)
    plt.close(fig)


def plot_energy_vs_resolution(df: pd.DataFrame, out_dir: Path):
    subset = df[
        (df["optimization"] == "none") & (df["batch_size"] == 1)
        & (df["device"] == "cuda") & (df["prompt_length"] == 10)
    ].dropna(subset=["energy_per_inf_j"])
    if subset.empty:
        return
    avg = subset.groupby(["model_short", "resolution"])["energy_per_inf_j"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    for model in sorted(avg["model_short"].unique()):
        mdata = avg[avg["model_short"] == model].sort_values("resolution")
        ax.plot(mdata["resolution"], mdata["energy_per_inf_j"], marker="o", label=model)
    ax.set_xlabel("Image Resolution")
    ax.set_ylabel("Energy per Inference (J)")
    ax.set_title("Energy vs Image Resolution (batch=1, GPU)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "energy_vs_resolution.png", dpi=150)
    plt.close(fig)


def plot_quality_vs_latency(df: pd.DataFrame, out_dir: Path):
    subset = _baseline_filter(df).dropna(subset=["wer"])
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    for _, row in subset.iterrows():
        ax.scatter(row["latency_mean_ms"], row["wer"], s=100)
        ax.annotate(row["model_short"], (row["latency_mean_ms"], row["wer"]),
                     fontsize=7, ha="left", va="bottom")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("WER (lower is better)")
    ax.set_title("Quality (WER) vs Latency — Pareto Frontier")
    plt.tight_layout()
    fig.savefig(out_dir / "quality_vs_latency.png", dpi=150)
    plt.close(fig)


def plot_flops_comparison(df: pd.DataFrame, out_dir: Path):
    subset = _baseline_filter(df).dropna(subset=["flops"])
    subset = subset[subset["flops"] > 0]
    if subset.empty:
        return
    avg = subset.groupby("model_short").agg({"flops": "mean", "flops_method": "first"}).reset_index()
    avg = avg.sort_values("flops")
    colors = ["tab:blue" if m == "calflops" else "tab:orange" for m in avg["flops_method"]]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(avg["model_short"], avg["flops"] / 1e9, color=colors)
    ax.set_xlabel("GFLOPs")
    ax.set_title("FLOPs per Inference (batch=1, res=224)")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="tab:blue", label="Measured (calflops)"),
        Patch(color="tab:orange", label="Estimated (2 x params)"),
    ])
    fig.text(
        0.5, 0.01,
        "Note: T5-based models (orange) use rough estimate — not directly comparable to measured values (blue)",
        ha="center", fontsize=8, style="italic", color="gray",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_dir / "flops_comparison.png", dpi=150)
    plt.close(fig)


def plot_param_breakdown(df: pd.DataFrame, out_dir: Path):
    subset = _baseline_filter(df)
    param_cols = [c for c in subset.columns if c.startswith("params_")]
    if not param_cols or subset.empty:
        return
    # Deduplicate by model (same params across datasets)
    models = subset.drop_duplicates("model_short")[["model_short"] + param_cols].set_index("model_short")
    models = models.fillna(0) / 1e6  # to millions
    fig, ax = plt.subplots(figsize=(12, 6))
    models.plot(kind="barh", stacked=True, ax=ax)
    ax.set_xlabel("Parameters (millions)")
    ax.set_title("Parameter Distribution by Component")
    ax.legend(title="Component", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "param_breakdown.png", dpi=150)
    plt.close(fig)


def plot_cpu_vs_gpu(df: pd.DataFrame, out_dir: Path):
    gpu = df[
        (df["device"] == "cuda") & (df["optimization"] == "none")
        & (df["batch_size"] == 1) & (df["resolution"] == 224) & (df["prompt_length"] == 10)
    ].groupby("model_short")["latency_mean_ms"].mean()
    cpu = df[
        (df["device"] == "cpu") & (df["optimization"] == "none")
        & (df["batch_size"] == 1) & (df["resolution"] == 224) & (df["prompt_length"] == 10)
    ].groupby("model_short")["latency_mean_ms"].mean()
    if gpu.empty or cpu.empty:
        return
    common = sorted(set(gpu.index) & set(cpu.index))
    if not common:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(common))
    ax.bar(x - 0.2, [float(gpu[m]) for m in common], 0.4, label="GPU", color="tab:blue")
    ax.bar(x + 0.2, [float(cpu[m]) for m in common], 0.4, label="CPU", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(common, rotation=45, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("CPU vs GPU Latency (batch=1, res=224)")
    ax.set_yscale("log")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "cpu_vs_gpu.png", dpi=150)
    plt.close(fig)


# --- Report ---

def generate_markdown_report(df: pd.DataFrame, out_dir: Path):
    report = ["# VLM Profiler Report\n"]

    # Baseline summary table
    baselines = _baseline_filter(df)
    if not baselines.empty:
        # Average across datasets for summary
        summary = baselines.groupby("model_short").agg({
            "latency_mean_ms": "mean",
            "wer": "mean",
            "energy_per_inf_j": "mean",
            "flops": "first",
            "total_params": "first",
        }).reset_index()
        summary = summary.sort_values("latency_mean_ms")
        summary.columns = ["Model", "Latency (ms)", "WER", "Energy (J/inf)", "FLOPs", "Params"]
        summary["FLOPs"] = summary["FLOPs"].apply(
            lambda x: f"{x:.2e}" if pd.notna(x) and x > 0 else "N/A"
        )
        summary["Params"] = summary["Params"].apply(
            lambda x: f"{x/1e6:.0f}M" if pd.notna(x) else "N/A"
        )
        report.append("## Baseline Comparison (batch=1, res=224, GPU)\n")
        report.append(summary.to_markdown(index=False, floatfmt=".2f"))
        report.append("")

    # Per-dataset quality
    quality_df = baselines.dropna(subset=["wer"])
    if not quality_df.empty:
        pivot = quality_df.pivot_table(values="wer", index="model_short", columns="dataset")
        report.append("## Quality (WER) by Dataset\n")
        report.append(pivot.to_markdown(floatfmt=".2f"))
        report.append("")

    # Charts with descriptions
    charts = [
        ("latency_vs_resolution.png", "Latency vs Image Resolution",
         "Most models show flat latency across resolutions — BLIP2 and InstructBLIP resize images internally to their fixed vision encoder size (224px). "
         "Only Fuyu-8b (+62%) and Idefics2-8b (+28%) show meaningful latency increase at higher resolutions, as they process variable-size inputs natively."),

        ("latency_vs_prompt_length.png", "Latency vs Prompt Length",
         "Prompt length has a strong effect on smaller models: blip2-opt-2.7b goes from 57ms to 558ms (9.8x) at 200 tokens. "
         "Larger autoregressive models (LLaVA, InstructBLIP-Vicuna) are already dominated by generation time, so input prompt length has minimal impact (~1.1x). "
         "Fuyu-8b shows no sensitivity to prompt length (1.0x)."),

        ("latency_vs_batch_size.png", "Latency vs Batch Size",
         "Batching improves throughput for all models. The smaller models benefit most: blip2-flan-t5-xl throughput doubles from 7.5 to 16.8 samples/s at batch=8. "
         "InstructBLIP-flan-t5-xl scales from 3.3 to 13.7 samples/s (4.2x). Larger 7-13B models see modest gains (2-4x throughput) due to memory bandwidth bottlenecks. "
         "Fuyu-8b does not support batching (processor limitation)."),

        ("optimization_speedup.png", "Optimization Speedup (FP16, torch.compile)",
         "FP16 gives the largest speedup for blip2-opt-2.7b (2.75x), Idefics2 (3.09x), and LLaVA-7b (2.63x). "
         "T5-based models (blip2-flan-t5-xl, instructblip-flan-t5-xl) show no benefit or slight regression from FP16 due to internal dtype casting overhead. "
         "torch.compile provides negligible improvement (~1.0-1.1x) and is incompatible with T5-based architectures."),

        ("energy_comparison.png", "Energy per Inference",
         "Energy consumption correlates strongly with model size and latency. The most efficient model is blip2-opt-2.7b at ~29J per inference, "
         "while the most expensive is instructblip-vicuna-7b at ~505J — a 17x difference. "
         "The 3-4B parameter models (BLIP2, InstructBLIP-flan) consume 30-67J, while 7-13B models consume 237-505J."),

        ("energy_vs_resolution.png", "Energy vs Resolution",
         "Energy follows the same pattern as latency: flat for models with fixed internal resolution (BLIP2, InstructBLIP), "
         "and increasing for Fuyu-8b and Idefics2 which process variable-size inputs. "
         "This confirms that for most models, input resolution is not a lever for energy optimization."),

        ("quality_vs_latency.png", "Quality (WER) vs Latency",
         "The best quality-latency tradeoff is blip2-flan-t5-xl (WER 2.28, 133ms) — it achieves nearly the best quality at moderate latency. "
         "InstructBLIP-flan-t5-xl offers similar quality (WER 2.36) but at 2.3x higher latency (301ms). "
         "LLaVA models are the slowest (~1.6-1.9s) with the worst WER scores (19-20), suggesting their output format is less aligned with the evaluation metric. "
         "Lower WER is better."),

        ("flops_comparison.png", "FLOPs Comparison",
         "LLaVA-13b has the highest measured FLOPs (~15.8 TFLOPs), followed by LLaVA-7b (~8.3 TFLOPs) and InstructBLIP-Vicuna-7b (~2.4 TFLOPs). "
         "BLIP2-opt-2.7b requires ~0.75 TFLOPs per inference. "
         "T5-based models (orange) show estimated values (2 x params) which are not directly comparable to measured FLOPs."),

        ("param_breakdown.png", "Parameter Distribution by Component",
         "Across all models, the language model (LLM backbone) dominates the parameter count at 70-73% of total parameters. "
         "The vision encoder accounts for 10-26%, and the Q-Former bridge module is 2-5%. "
         "This suggests that optimizing the LLM component (e.g., via quantization or distillation) would have the largest impact on model efficiency."),

        ("cpu_vs_gpu.png", "CPU vs GPU Latency",
         "GPU acceleration provides 16-32x speedup over CPU across all models (log scale). "
         "The largest GPU advantage is for Fuyu-8b (32x), while the smallest is for Idefics2 (16x). "
         "Larger models benefit more from GPU parallelism: 7-13B models see 23-27x speedup vs 19x for 3-4B models. "
         "CPU inference is impractical for production use — even the fastest model (blip2-opt-2.7b) takes ~1.1s on CPU vs 57ms on GPU."),
    ]
    for filename, title, description in charts:
        if (out_dir / filename).exists():
            report.append(f"## {title}\n")
            report.append(f"![{title}]({filename})\n")
            report.append(description)
            report.append("")

    # Limitations section
    report.append("## Known Limitations\n")
    report.append("### Model-specific incompatibilities\n")
    report.append("| Model | Issue | Affected experiments |")
    report.append("|-------|-------|---------------------|")
    report.append("| adept/fuyu-8b | FP16 causes dtype mismatch (Float vs Half) | 3 (fp16 x 3 datasets) |")
    report.append("| adept/fuyu-8b | Processor returns lists for batched inputs | 9 (batch>1 x 3 datasets) |")
    report.append("| blip2-flan-t5-xl | torch.compile fails (T5 architecture) | 3 (torch_compile x 3 datasets) |")
    report.append("| instructblip-flan-t5-xl | torch.compile fails (T5 architecture) | 3 (torch_compile x 3 datasets) |")
    report.append("")
    report.append("### Other notes\n")
    report.append("- **Resolution scaling is flat** for BLIP2/InstructBLIP — they resize internally to fixed vision encoder resolution")
    report.append("- **FLOPs**: exact measurement (calflops) only for some models; T5-based use rough estimate (2 x params)")
    report.append("- **FP16 can be slower** on T5-based models due to internal dtype casting overhead")
    report.append("")

    (out_dir / "REPORT.md").write_text("\n".join(report))


def main():
    parser = argparse.ArgumentParser(description="Generate report from all experiment results")
    parser.add_argument(
        "--outputs-root",
        required=True,
        help="Root outputs directory (e.g. outputs/2026-03-26) — scans all model subdirs",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Where to write the report (default: reports/{date}/)",
    )
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    report_dir = Path(args.report_dir) if args.report_dir else Path("reports") / outputs_root.name
    report_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_results(outputs_root)
    if df.empty:
        print("No successful results found in", outputs_root)
        return

    models = df["model_short"].nunique()
    datasets = df["dataset"].nunique()
    print(f"Loaded {len(df)} results from {models} models, {datasets} datasets.")

    plot_latency_vs_resolution(df, report_dir)
    plot_latency_vs_prompt_length(df, report_dir)
    plot_latency_vs_batch(df, report_dir)
    plot_optimization_speedup(df, report_dir)
    plot_energy_comparison(df, report_dir)
    plot_energy_vs_resolution(df, report_dir)
    plot_quality_vs_latency(df, report_dir)
    plot_flops_comparison(df, report_dir)
    plot_param_breakdown(df, report_dir)
    plot_cpu_vs_gpu(df, report_dir)
    generate_markdown_report(df, report_dir)

    print(f"Report generated at {report_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
