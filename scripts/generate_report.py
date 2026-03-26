#!/usr/bin/env python3
"""Aggregate experiment results and generate charts + markdown report."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load all JSON result files into a DataFrame."""
    records = []
    for f in results_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            if data.get("status") != "success":
                continue
            flat = {
                "model": data["model"],
                "dataset": data["dataset"],
                "resolution": data["resolution"],
                "prompt_length": data["prompt_length"],
                "device": data["device"],
                "optimization": data["optimization"],
                "batch_size": data["batch_size"],
                "latency_mean_ms": data["latency"]["mean_ms"],
                "latency_p50_ms": data["latency"]["p50_ms"],
                "latency_p95_ms": data["latency"]["p95_ms"],
                "wer": data["quality"]["wer_score"],
                "exact_match": data["quality"]["exact_match_accuracy"],
            }
            if data.get("energy"):
                flat["avg_power_w"] = data["energy"]["avg_power_w"]
                flat["energy_j"] = data["energy"]["energy_j"]
            else:
                flat["avg_power_w"] = None
                flat["energy_j"] = None
            records.append(flat)
        except (json.JSONDecodeError, KeyError):
            continue
    return pd.DataFrame(records)


def plot_latency_vs_resolution(df: pd.DataFrame, out_dir: Path):
    """Latency vs resolution for each model (baseline config)."""
    subset = df[
        (df["optimization"] == "none")
        & (df["batch_size"] == 1)
        & (df["device"] == "cuda")
        & (df["prompt_length"] == 10)
    ]
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for model in subset["model"].unique():
        mdata = subset[subset["model"] == model].sort_values("resolution")
        short_name = model.split("/")[-1]
        ax.plot(mdata["resolution"], mdata["latency_mean_ms"], marker="o", label=short_name)
    ax.set_xlabel("Image Resolution")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency vs Image Resolution (batch=1, FP32, GPU)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "latency_vs_resolution.png", dpi=150)
    plt.close(fig)


def plot_latency_vs_batch(df: pd.DataFrame, out_dir: Path):
    """Latency vs batch size."""
    subset = df[
        (df["optimization"] == "none")
        & (df["resolution"] == 224)
        & (df["device"] == "cuda")
        & (df["prompt_length"] == 10)
    ]
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for model in subset["model"].unique():
        mdata = subset[subset["model"] == model].sort_values("batch_size")
        short_name = model.split("/")[-1]
        ax.plot(mdata["batch_size"], mdata["latency_mean_ms"], marker="s", label=short_name)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency vs Batch Size (res=224, FP32, GPU)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "latency_vs_batch_size.png", dpi=150)
    plt.close(fig)


def plot_optimization_speedup(df: pd.DataFrame, out_dir: Path):
    """Speedup from FP16/compile/FA2 relative to FP32 baseline."""
    baseline = df[
        (df["optimization"] == "none")
        & (df["batch_size"] == 1)
        & (df["device"] == "cuda")
        & (df["resolution"] == 224)
        & (df["prompt_length"] == 10)
    ].set_index("model")["latency_mean_ms"]

    if baseline.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for opt in ["fp16", "torch_compile", "flash_attn2"]:
        opt_data = df[
            (df["optimization"] == opt)
            & (df["batch_size"] == 1)
            & (df["device"] == "cuda")
            & (df["resolution"] == 224)
            & (df["prompt_length"] == 10)
        ].set_index("model")["latency_mean_ms"]

        speedups = baseline / opt_data
        speedups = speedups.dropna()
        if not speedups.empty:
            short_names = [m.split("/")[-1] for m in speedups.index]
            ax.bar(
                [f"{n}" for n in short_names],
                speedups.values,
                alpha=0.7,
                label=opt,
            )
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Speedup (vs FP32)")
    ax.set_title("Optimization Speedup per Model")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_dir / "optimization_speedup.png", dpi=150)
    plt.close(fig)


def plot_quality_vs_latency(df: pd.DataFrame, out_dir: Path):
    """Pareto frontier: quality vs latency."""
    subset = df[
        (df["optimization"] == "none")
        & (df["batch_size"] == 1)
        & (df["device"] == "cuda")
        & (df["resolution"] == 224)
        & (df["prompt_length"] == 10)
    ]
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    for _, row in subset.iterrows():
        short_name = row["model"].split("/")[-1]
        ax.scatter(row["latency_mean_ms"], row["exact_match"], s=100)
        ax.annotate(short_name, (row["latency_mean_ms"], row["exact_match"]),
                     fontsize=7, ha="left", va="bottom")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Exact Match Accuracy")
    ax.set_title("Quality vs Latency (Pareto Frontier)")
    plt.tight_layout()
    fig.savefig(out_dir / "quality_vs_latency.png", dpi=150)
    plt.close(fig)


def plot_energy_comparison(df: pd.DataFrame, out_dir: Path):
    """Energy consumption per model."""
    subset = df[
        (df["optimization"] == "none")
        & (df["batch_size"] == 1)
        & (df["device"] == "cuda")
        & (df["resolution"] == 224)
        & (df["prompt_length"] == 10)
        & (df["energy_j"].notna())
    ]
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    short_names = [m.split("/")[-1] for m in subset["model"]]
    ax.barh(short_names, subset["energy_j"])
    ax.set_xlabel("Energy (J)")
    ax.set_title("Energy Consumption per Inference (batch=1, res=224)")
    plt.tight_layout()
    fig.savefig(out_dir / "energy_comparison.png", dpi=150)
    plt.close(fig)


def generate_markdown_report(df: pd.DataFrame, out_dir: Path):
    """Generate a markdown report with embedded chart references."""
    report = ["# VLM Profiler Report\n"]

    # Summary table
    summary = df[
        (df["optimization"] == "none")
        & (df["batch_size"] == 1)
        & (df["device"] == "cuda")
        & (df["resolution"] == 224)
        & (df["prompt_length"] == 10)
    ][["model", "latency_mean_ms", "exact_match", "wer", "energy_j"]].copy()

    if not summary.empty:
        summary["model"] = summary["model"].apply(lambda x: x.split("/")[-1])
        summary = summary.sort_values("latency_mean_ms")
        report.append("## Baseline Comparison (batch=1, res=224, FP32, GPU)\n")
        report.append(summary.to_markdown(index=False))
        report.append("")

    # Charts
    charts = [
        ("latency_vs_resolution.png", "Latency vs Image Resolution"),
        ("latency_vs_batch_size.png", "Latency vs Batch Size"),
        ("optimization_speedup.png", "Optimization Speedup"),
        ("quality_vs_latency.png", "Quality vs Latency (Pareto)"),
        ("energy_comparison.png", "Energy Consumption"),
    ]
    for filename, title in charts:
        if (out_dir / filename).exists():
            report.append(f"## {title}\n")
            report.append(f"![{title}]({filename})\n")

    (out_dir / "REPORT.md").write_text("\n".join(report))


def main():
    parser = argparse.ArgumentParser(description="Generate report from experiment results")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Experiment output directory (e.g. outputs/2026-03-26/exp_name/12-00-07)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results_dir = output_dir / "results"
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(results_dir)
    if df.empty:
        print("No successful results found in", results_dir)
        return

    print(f"Loaded {len(df)} successful experiment results.")

    plot_latency_vs_resolution(df, reports_dir)
    plot_latency_vs_batch(df, reports_dir)
    plot_optimization_speedup(df, reports_dir)
    plot_quality_vs_latency(df, reports_dir)
    plot_energy_comparison(df, reports_dir)
    generate_markdown_report(df, reports_dir)

    print(f"Report generated at {reports_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
