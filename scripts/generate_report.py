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
    """Scan outputs_root/{model_name}/{timestamp}/results/*.json and merge.

    When duplicate experiment configs exist (e.g. from reruns), keeps only
    the latest result based on file path (which encodes date/timestamp).
    """
    # Collect all results keyed by experiment config, later file wins
    by_config: dict[tuple, dict] = {}
    for json_file in sorted(outputs_root.rglob("results/*.json")):
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
            # Dedup key: keep latest result per experiment config
            config_key = (
                flat["model"], flat["dataset"], flat["resolution"],
                flat["prompt_length"], flat["device"],
                flat["optimization"], flat["batch_size"],
            )
            by_config[config_key] = flat
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    records = list(by_config.values())
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


def _model_colors(models: list[str]) -> dict:
    """Assign consistent colors to models sorted by latency."""
    cmap = plt.cm.get_cmap("tab10", len(models))
    return {m: cmap(i) for i, m in enumerate(models)}


def _plot_wer_latency_single(
    ax, data: pd.DataFrame, color_map: dict, title: str, show_legend: bool = False,
):
    """Plot WER vs latency on a single axes with Pareto frontier and labels."""
    from adjustText import adjust_text

    for _, row in data.iterrows():
        m = row["model_short"]
        ax.scatter(
            row["latency_mean_ms"], row["wer"],
            s=140, color=color_map.get(m, "gray"), edgecolors="black",
            linewidths=0.7, zorder=5, label=m if show_legend else None,
        )

    # Pareto frontier
    pareto = data.sort_values("latency_mean_ms")
    frontier = []
    best_wer = float("inf")
    for _, row in pareto.iterrows():
        if row["wer"] < best_wer:
            frontier.append(row)
            best_wer = row["wer"]
    if len(frontier) > 1:
        ax.plot(
            [r["latency_mean_ms"] for r in frontier],
            [r["wer"] for r in frontier],
            "--", color="gray", alpha=0.5, linewidth=1.5,
        )

    # Labels
    texts = []
    for _, row in data.iterrows():
        texts.append(ax.text(
            row["latency_mean_ms"], row["wer"], row["model_short"],
            fontsize=7.5, fontweight="bold",
        ))
    adjust_text(
        texts, ax=ax,
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        expand=(1.8, 1.8),
        force_text=(1.2, 1.2),
        force_points=(0.8, 0.8),
    )

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Latency (ms)", fontsize=9)
    ax.set_ylabel("WER (lower is better)", fontsize=9)
    wer_min, wer_max = data["wer"].min(), data["wer"].max()
    wer_range = wer_max - wer_min if wer_max > wer_min else wer_max * 0.2
    padding = max(wer_range * 0.3, 0.1)
    ax.set_ylim(bottom=max(0, wer_min - padding), top=wer_max + padding)


def plot_quality_vs_latency(df: pd.DataFrame, out_dir: Path):
    subset = _baseline_filter(df).dropna(subset=["wer"])
    if subset.empty:
        return

    # Average across datasets for summary
    avg = (
        subset.groupby("model_short")
        .agg(latency_mean_ms=("latency_mean_ms", "mean"), wer=("wer", "mean"))
        .reset_index()
    )
    models_sorted = avg.sort_values("latency_mean_ms")["model_short"].tolist()
    color_map = _model_colors(models_sorted)

    datasets = sorted(subset["dataset"].unique())

    # --- Per-dataset charts (one subplot per dataset) ---
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5.5))
    if len(datasets) == 1:
        axes = [axes]
    dataset_titles = {"scienceqa": "ScienceQA", "textvqa": "TextVQA", "coco_caption": "COCO Caption"}
    for ax_i, ds in zip(axes, datasets):
        ds_data = subset[subset["dataset"] == ds].copy()
        _plot_wer_latency_single(ax_i, ds_data, color_map, dataset_titles.get(ds, ds))

    fig.suptitle("Quality (WER) vs Latency — by Dataset", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "quality_vs_latency_by_dataset.png", dpi=150, bbox_inches="tight")
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


def plot_component_latency_breakdown(breakdown_path: Path, out_dir: Path):
    """Stacked horizontal bar chart of per-component latency share per model.

    Reads results/component_breakdown.json produced by scripts/component_breakdown.py.
    """
    if not breakdown_path.exists():
        return
    try:
        data = json.loads(breakdown_path.read_text())
    except json.JSONDecodeError:
        return

    records = []
    for rec in data.get("results", []):
        if "component_times" not in rec:
            continue
        per_call = rec["component_times"].get("per_call_ms", {})
        if not per_call:
            continue
        model_short = rec["model"].split("/")[-1]
        row = {"model_short": model_short, **per_call}
        row["_total"] = sum(per_call.values())
        records.append(row)
    if not records:
        return
    df_comp = pd.DataFrame(records).fillna(0.0)
    df_comp = df_comp.sort_values("_total")
    component_cols = [c for c in df_comp.columns if c not in ("model_short", "_total")]

    # Absolute ms chart
    fig, ax = plt.subplots(figsize=(12, 5.5))
    left = np.zeros(len(df_comp))
    color_palette = plt.cm.get_cmap("tab10", len(component_cols))
    for i, comp in enumerate(component_cols):
        vals = df_comp[comp].values
        ax.barh(df_comp["model_short"], vals, left=left, label=comp, color=color_palette(i))
        left = left + vals
    ax.set_xlabel("Latency per inference (ms, sum of forward-pass time per submodule)")
    ax.set_title("Per-component latency breakdown (baseline: res=224, plen=10, bs=1, FP32)")
    ax.legend(title="Component", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "component_latency_breakdown.png", dpi=150)
    plt.close(fig)

    # Share chart
    fig, ax = plt.subplots(figsize=(12, 5.5))
    shares = df_comp[component_cols].div(df_comp[component_cols].sum(axis=1), axis=0) * 100.0
    left = np.zeros(len(df_comp))
    for i, comp in enumerate(component_cols):
        vals = shares[comp].values
        ax.barh(df_comp["model_short"], vals, left=left, label=comp, color=color_palette(i))
        left = left + vals
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of measured component time (%)")
    ax.set_title("Component share of inference time (baseline: res=224, plen=10, bs=1, FP32)")
    ax.legend(title="Component", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "component_latency_share.png", dpi=150)
    plt.close(fig)


def plot_prefill_vs_decode(breakdown_path: Path, out_dir: Path):
    """Grouped bar chart: prefill cost vs decode cost per token."""
    if not breakdown_path.exists():
        return
    try:
        data = json.loads(breakdown_path.read_text())
    except json.JSONDecodeError:
        return

    rows = []
    for rec in data.get("results", []):
        pd_times = rec.get("prefill_decode")
        if not pd_times:
            continue
        rows.append({
            "model_short": rec["model"].split("/")[-1],
            "prefill_ms": pd_times["prefill_ms"],
            "decode_ms_per_token": pd_times["decode_per_token_ms"],
        })
    if not rows:
        return
    df_pd = pd.DataFrame(rows).sort_values("prefill_ms")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].barh(df_pd["model_short"], df_pd["prefill_ms"], color="tab:blue")
    axes[0].set_xlabel("Prefill latency (ms)")
    axes[0].set_title("Prefill cost per inference")
    axes[1].barh(df_pd["model_short"], df_pd["decode_ms_per_token"], color="tab:green")
    axes[1].set_xlabel("Decode latency per token (ms)")
    axes[1].set_title("Per-token decode cost")
    plt.tight_layout()
    fig.savefig(out_dir / "prefill_vs_decode.png", dpi=150)
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

    # Charts
    charts = [
        ("latency_vs_resolution.png", "Latency vs Image Resolution"),
        ("latency_vs_prompt_length.png", "Latency vs Prompt Length"),
        ("latency_vs_batch_size.png", "Latency vs Batch Size"),
        ("optimization_speedup.png", "Optimization Speedup (FP16, torch.compile)"),
        ("energy_comparison.png", "Energy per Inference"),
        ("energy_vs_resolution.png", "Energy vs Resolution"),
        ("quality_vs_latency_by_dataset.png", "Quality (WER) vs Latency — by Dataset"),
        ("flops_comparison.png", "FLOPs Comparison"),
        ("param_breakdown.png", "Parameter Distribution by Component"),
        ("cpu_vs_gpu.png", "CPU vs GPU Latency"),
        ("component_latency_share.png", "Per-Component Share of Inference Time"),
        ("component_latency_breakdown.png", "Per-Component Latency Breakdown (ms)"),
        ("prefill_vs_decode.png", "Prefill vs Per-Token Decode Cost"),
    ]
    for filename, title in charts:
        if (out_dir / filename).exists():
            report.append(f"## {title}\n")
            report.append(f"![{title}]({filename})\n")

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

    # Component breakdown charts (read from a separate JSON)
    breakdown_path = Path("results/component_breakdown.json")
    plot_component_latency_breakdown(breakdown_path, report_dir)
    plot_prefill_vs_decode(breakdown_path, report_dir)

    generate_markdown_report(df, report_dir)

    print(f"Report generated at {report_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
