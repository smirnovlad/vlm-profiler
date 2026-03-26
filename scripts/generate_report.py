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
    fig, ax = plt.subplots(figsize=(12, 6))
    for model in sorted(subset["model_short"].unique()):
        mdata = subset[subset["model_short"] == model].sort_values("resolution")
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
    fig, ax = plt.subplots(figsize=(12, 6))
    for model in sorted(subset["model_short"].unique()):
        mdata = subset[subset["model_short"] == model].sort_values("prompt_length")
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
    fig, ax = plt.subplots(figsize=(12, 6))
    for model in sorted(subset["model_short"].unique()):
        mdata = subset[subset["model_short"] == model].sort_values("batch_size")
        ax.plot(mdata["batch_size"], mdata["latency_mean_ms"], marker="s", label=model)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency vs Batch Size (res=224, GPU)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "latency_vs_batch_size.png", dpi=150)
    plt.close(fig)


def plot_optimization_speedup(df: pd.DataFrame, out_dir: Path):
    base = _baseline_filter(df).set_index("model_short")["latency_mean_ms"]
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
        ].set_index("model_short")["latency_mean_ms"]
        speedups = [base.get(m, None) / opt_data.get(m, None) if m in opt_data else 0 for m in models]
        ax.bar(x + i * width, speedups, width, label=opt, alpha=0.8)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Speedup (vs baseline)")
    ax.set_title("Optimization Speedup per Model")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "optimization_speedup.png", dpi=150)
    plt.close(fig)


def plot_energy_comparison(df: pd.DataFrame, out_dir: Path):
    subset = _baseline_filter(df).dropna(subset=["energy_per_inf_j"])
    if subset.empty:
        return
    subset = subset.sort_values("energy_per_inf_j")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(subset["model_short"], subset["energy_per_inf_j"])
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
    fig, ax = plt.subplots(figsize=(12, 6))
    for model in sorted(subset["model_short"].unique()):
        mdata = subset[subset["model_short"] == model].sort_values("resolution")
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
    subset = subset.sort_values("flops")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(subset["model_short"], subset["flops"] / 1e9)
    ax.set_xlabel("GFLOPs")
    ax.set_title("FLOPs per Inference (batch=1, res=224)")
    plt.tight_layout()
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
    ].set_index("model_short")["latency_mean_ms"]
    cpu = df[
        (df["device"] == "cpu") & (df["optimization"] == "none")
        & (df["batch_size"] == 1) & (df["resolution"] == 224) & (df["prompt_length"] == 10)
    ].set_index("model_short")["latency_mean_ms"]
    if gpu.empty or cpu.empty:
        return
    common = sorted(set(gpu.index) & set(cpu.index))
    if not common:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(common))
    ax.bar(x - 0.2, [gpu[m] for m in common], 0.4, label="GPU", color="tab:blue")
    ax.bar(x + 0.2, [cpu[m] for m in common], 0.4, label="CPU", color="tab:orange")
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

    # Charts
    charts = [
        ("latency_vs_resolution.png", "Latency vs Image Resolution"),
        ("latency_vs_prompt_length.png", "Latency vs Prompt Length"),
        ("latency_vs_batch_size.png", "Latency vs Batch Size"),
        ("optimization_speedup.png", "Optimization Speedup (FP16, torch.compile)"),
        ("energy_comparison.png", "Energy per Inference"),
        ("energy_vs_resolution.png", "Energy vs Resolution"),
        ("quality_vs_latency.png", "Quality (WER) vs Latency"),
        ("flops_comparison.png", "FLOPs Comparison"),
        ("param_breakdown.png", "Parameter Distribution by Component"),
        ("cpu_vs_gpu.png", "CPU vs GPU Latency"),
    ]
    for filename, title in charts:
        if (out_dir / filename).exists():
            report.append(f"## {title}\n")
            report.append(f"![{title}]({filename})\n")

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
