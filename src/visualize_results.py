# src/results_viz.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# =============================================================================
# Loading / setup
# =============================================================================

def load_core_results(results_dir: str | Path) -> pd.DataFrame:
    """
    Load core scenario results saved by run_core_scenarios().

    Expected file:
        <results_dir>/core_scenario_results.csv
    """
    results_dir = Path(results_dir)
    df = pd.read_csv(results_dir / "core_scenario_results.csv")

    if "renewable_correlation_case" in df.columns:
        df["renewable_correlation_case"] = pd.Categorical(
            df["renewable_correlation_case"],
            categories=["low", "high"],
            ordered=True,
        )

    if "ldes_duration_hours" in df.columns:
        df["ldes_duration_hours"] = pd.to_numeric(
            df["ldes_duration_hours"], errors="coerce"
        )

    return df.sort_values(
        ["renewable_correlation_case", "ldes_duration_hours"]
    ).reset_index(drop=True)


def check_columns(df: pd.DataFrame) -> None:
    """
    Print available columns and a small preview.
    """
    print("Columns:")
    print(sorted(df.columns.tolist()))
    print("\nPreview:")
    print(df.head())

    expected = [
        "scenario_id",
        "renewable_correlation_case",
        "ldes_duration_hours",
        "ldes_mw",
        "ldes_mwh",
        "battery4h_mw",
        "transmission_mw",
        "system_cost_per_mwh",
        "vres_curtailment_mwh",
    ]

    missing = [c for c in expected if c not in df.columns]
    if missing:
        print("\nMissing expected columns:")
        print(missing)
    else:
        print("\nAll expected core columns are present.")


def make_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a compact summary table for inspection/export.
    """
    cols = [
        "scenario_id",
        "renewable_correlation_case",
        "ldes_duration_hours",
        "ldes_mw",
        "ldes_mwh",
        "battery4h_mw",
        "transmission_mw",
        "backup_mw",
        "solar_mw",
        "wind_mw",
        "system_cost_per_mwh",
        "vres_curtailment_mwh",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()


# Style

def set_paper_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.figsize": (7, 4.5),
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,
        "lines.linewidth": 2.4,
        "lines.markersize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def set_slide_style() -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.15)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.figsize": (9, 5.5),
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "legend.title_fontsize": 12,
        "lines.linewidth": 3.0,
        "lines.markersize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# Helpers

def _duration_ticks(results: pd.DataFrame) -> list[int]:
    vals = results["ldes_duration_hours"].dropna().astype(int).unique().tolist()
    return sorted(vals)


def _clean_axes(ax: plt.Axes, results: pd.DataFrame) -> None:
    ax.set_xticks(_duration_ticks(results))
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(False, axis="x")


def save_figure(
    filename: str,
    folder: str | Path = "figures",
    close: bool = False,
) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    plt.savefig(path, bbox_inches="tight")
    if close:
        plt.close()
    return path


def _scenario_labels(results: pd.DataFrame) -> pd.Series:
    return (
        results["renewable_correlation_case"].astype(str)
        + "_"
        + results["ldes_duration_hours"].astype(int).astype(str)
        + "h"
    )


def _basic_lineplot(
    results: pd.DataFrame,
    y: str,
    title: str,
    ylabel: str,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()

    sns.lineplot(
        data=results,
        x="ldes_duration_hours",
        y=y,
        hue="renewable_correlation_case",
        style="renewable_correlation_case",
        marker="o",
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("LDES Duration (hours)")
    ax.set_ylabel(ylabel)
    _clean_axes(ax, results)
    ax.legend(title="Correlation Case", frameon=True)
    return ax


# Core plots

def plot_system_cost(results: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    return _basic_lineplot(
        results,
        y="system_cost_per_mwh",
        title="System Cost by LDES Duration",
        ylabel="System Cost ($/MWh)",
        ax=ax,
    )


def plot_curtailment(results: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    return _basic_lineplot(
        results,
        y="vres_curtailment_mwh",
        title="Renewable Curtailment by LDES Duration",
        ylabel="Curtailment (MWh)",
        ax=ax,
    )


def plot_transmission_build(results: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    return _basic_lineplot(
        results,
        y="transmission_mw",
        title="Transmission Capacity by LDES Duration",
        ylabel="Transmission Capacity (MW)",
        ax=ax,
    )


def plot_ldes_power(results: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    return _basic_lineplot(
        results,
        y="ldes_mw",
        title="LDES Power Capacity by Duration",
        ylabel="LDES Capacity (MW)",
        ax=ax,
    )


def plot_ldes_energy(results: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    return _basic_lineplot(
        results,
        y="ldes_mwh",
        title="LDES Energy Capacity by Duration",
        ylabel="LDES Energy Capacity (MWh)",
        ax=ax,
    )


def plot_battery_power(results: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    return _basic_lineplot(
        results,
        y="battery4h_mw",
        title="Battery Capacity by LDES Duration",
        ylabel="Battery Capacity (MW)",
        ax=ax,
    )


def plot_backup_capacity(results: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    return _basic_lineplot(
        results,
        y="backup_mw",
        title="Backup Capacity by LDES Duration",
        ylabel="Backup Capacity (MW)",
        ax=ax,
    )


def plot_wind_build(results: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    return _basic_lineplot(
        results,
        y="wind_mw",
        title="Wind Capacity by LDES Duration",
        ylabel="Wind Capacity (MW)",
        ax=ax,
    )


def plot_solar_build(results: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    return _basic_lineplot(
        results,
        y="solar_mw",
        title="Solar Capacity by LDES Duration",
        ylabel="Solar Capacity (MW)",
        ax=ax,
    )


def plot_storage_comparison(results: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    needed = ["battery4h_mw", "ldes_mw"]
    missing = [c for c in needed if c not in results.columns]
    if missing:
        raise ValueError(f"Missing columns for storage comparison: {missing}")

    df = results.copy()
    df["scenario_label"] = _scenario_labels(df)

    long_df = df.melt(
        id_vars=["scenario_label"],
        value_vars=["battery4h_mw", "ldes_mw"],
        var_name="storage_type",
        value_name="capacity_mw",
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 4.8))

    sns.barplot(
        data=long_df,
        x="scenario_label",
        y="capacity_mw",
        hue="storage_type",
        ax=ax,
    )

    ax.set_title("Short- and Long-Duration Storage Deployment")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Installed Capacity (MW)")
    ax.legend(title="Storage Type", frameon=True)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(False, axis="x")
    return ax


def plot_ldes_vs_transmission(results: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    sns.scatterplot(
        data=results,
        x="transmission_mw",
        y="ldes_mw",
        hue="renewable_correlation_case",
        style="ldes_duration_hours",
        s=140,
        ax=ax,
    )

    ax.set_title("LDES vs Transmission Deployment")
    ax.set_xlabel("Transmission Capacity (MW)")
    ax.set_ylabel("LDES Capacity (MW)")
    ax.legend(title="Scenario", frameon=True)
    return ax


def plot_cost_components(results: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    needed = [
        "generator_capex_total",
        "storage_capex_total",
        "link_capex_total",
        "generator_opex_total",
    ]
    missing = [c for c in needed if c not in results.columns]
    if missing:
        raise ValueError(f"Missing cost component columns: {missing}")

    df = results.copy()
    df["scenario_label"] = _scenario_labels(df)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    x = range(len(df))
    bottom = None

    components = [
        ("generator_capex_total", "Generator CAPEX"),
        ("storage_capex_total", "Storage CAPEX"),
        ("link_capex_total", "Transmission CAPEX"),
        ("generator_opex_total", "Generator OPEX"),
    ]

    for col, label in components:
        values = df[col].values
        if bottom is None:
            ax.bar(x, values, label=label)
            bottom = values.copy()
        else:
            ax.bar(x, values, bottom=bottom, label=label)
            bottom = bottom + values

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["scenario_label"], rotation=30)
    ax.set_title("Cost Components by Scenario")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Annualized Cost ($)")
    ax.legend(frameon=True)
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(False, axis="x")
    return ax


def plot_capex_vs_opex(results: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    needed = ["total_capex", "total_opex"]
    missing = [c for c in needed if c not in results.columns]
    if missing:
        raise ValueError(f"Missing columns for CAPEX/OPEX comparison: {missing}")

    df = results.copy()
    df["scenario_label"] = _scenario_labels(df)

    long_df = df.melt(
        id_vars=["scenario_label"],
        value_vars=["total_capex", "total_opex"],
        var_name="cost_type",
        value_name="cost",
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 4.8))

    sns.barplot(
        data=long_df,
        x="scenario_label",
        y="cost",
        hue="cost_type",
        ax=ax,
    )

    ax.set_title("CAPEX and OPEX by Scenario")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Annualized Cost ($)")
    ax.legend(title="Cost Type", frameon=True)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(False, axis="x")
    return ax


def plot_cost_reconstruction_error(results: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    if "cost_reconstruction_error" not in results.columns:
        raise ValueError("Missing column: cost_reconstruction_error")

    df = results.copy()
    df["scenario_label"] = _scenario_labels(df)

    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 4.5))

    sns.barplot(
        data=df,
        x="scenario_label",
        y="cost_reconstruction_error",
        ax=ax,
    )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Cost Reconstruction Error")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Objective - Reconstructed Cost ($)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(False, axis="x")
    return ax


def plot_generation_mix_bars(results: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    needed = ["wind_mw", "solar_mw", "backup_mw"]
    missing = [c for c in needed if c not in results.columns]
    if missing:
        raise ValueError(f"Missing columns for generation mix: {missing}")

    df = results.copy()
    df["scenario_label"] = _scenario_labels(df)

    long_df = df.melt(
        id_vars=["scenario_label"],
        value_vars=["wind_mw", "solar_mw", "backup_mw"],
        var_name="generation_type",
        value_name="capacity_mw",
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=long_df,
        x="scenario_label",
        y="capacity_mw",
        hue="generation_type",
        ax=ax,
    )

    ax.set_title("Generation Capacity Mix by Scenario")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Installed Capacity (MW)")
    ax.legend(title="Generation Type", frameon=True)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(False, axis="x")
    return ax


def plot_longform_facets(
    results: pd.DataFrame,
    metrics: Iterable[str],
    metric_labels: dict[str, str] | None = None,
):
    """
    Quick faceted comparison plot for multiple metrics.
    """
    metrics = list(metrics)
    missing = [m for m in metrics if m not in results.columns]
    if missing:
        raise ValueError(f"Missing metric columns: {missing}")

    df = results.copy()
    long_df = df.melt(
        id_vars=["renewable_correlation_case", "ldes_duration_hours"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )

    if metric_labels is not None:
        long_df["metric"] = long_df["metric"].replace(metric_labels)

    g = sns.relplot(
        data=long_df,
        x="ldes_duration_hours",
        y="value",
        hue="renewable_correlation_case",
        style="renewable_correlation_case",
        kind="line",
        marker="o",
        col="metric",
        col_wrap=2,
        facet_kws={"sharey": False},
        height=4,
        aspect=1.3,
    )

    for ax in g.axes.flat:
        ax.set_xticks(_duration_ticks(results))
        ax.grid(True, axis="y", alpha=0.25)
        ax.grid(False, axis="x")

    g.set_xlabels("LDES Duration (hours)")
    g.set_ylabels("")
    return g


# =============================================================================
# Multi-panel figure sets
# =============================================================================

def plot_paper_main_panel(results: pd.DataFrame):
    """
    Main 3-panel figure for the paper.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plot_system_cost(results, ax=axes[0])
    plot_transmission_build(results, ax=axes[1])
    plot_curtailment(results, ax=axes[2])

    axes[0].set_title("A. System Cost")
    axes[1].set_title("B. Transmission Capacity")
    axes[2].set_title("C. Curtailment")

    fig.tight_layout()
    return fig, axes


def plot_paper_alt_panel(results: pd.DataFrame):
    """
    Alternative paper panel emphasizing storage instead of curtailment.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plot_system_cost(results, ax=axes[0])
    plot_transmission_build(results, ax=axes[1])
    plot_ldes_power(results, ax=axes[2])

    axes[0].set_title("A. System Cost")
    axes[1].set_title("B. Transmission Capacity")
    axes[2].set_title("C. LDES Capacity")

    fig.tight_layout()
    return fig, axes


def plot_slide_main_panel(results: pd.DataFrame):
    """
    Cleaner 2-panel figure for slides.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_system_cost(results, ax=axes[0])
    plot_ldes_power(results, ax=axes[1])

    axes[0].set_title("System Cost")
    axes[1].set_title("LDES Deployment")

    fig.tight_layout()
    return fig, axes


# =============================================================================
# Batch creators
# =============================================================================

def create_paper_figures(results: pd.DataFrame, outdir: str | Path = "figures/paper") -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    set_paper_style()

    fig, _ = plot_paper_main_panel(results)
    fig.savefig(outdir / "paper_main_panel.png", bbox_inches="tight")
    plt.close(fig)

    fig, _ = plot_paper_alt_panel(results)
    fig.savefig(outdir / "paper_alt_panel.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    plot_system_cost(results, ax=ax)
    fig.savefig(outdir / "paper_system_cost.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    plot_storage_comparison(results, ax=ax)
    fig.savefig(outdir / "paper_storage_comparison.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    plot_ldes_vs_transmission(results, ax=ax)
    fig.savefig(outdir / "paper_ldes_vs_transmission.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    plot_ldes_energy(results, ax=ax)
    fig.savefig(outdir / "paper_ldes_energy.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    plot_backup_capacity(results, ax=ax)
    fig.savefig(outdir / "paper_backup_capacity.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_generation_mix_bars(results, ax=ax)
    fig.savefig(outdir / "paper_generation_mix.png", bbox_inches="tight")
    plt.close(fig)

    cost_cols = {
        "generator_capex_total",
        "storage_capex_total",
        "link_capex_total",
        "generator_opex_total",
        "total_capex",
        "total_opex",
    }
    if cost_cols.issubset(results.columns):
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_cost_components(results, ax=ax)
        fig.savefig(outdir / "paper_cost_components.png", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        plot_capex_vs_opex(results, ax=ax)
        fig.savefig(outdir / "paper_capex_vs_opex.png", bbox_inches="tight")
        plt.close(fig)

    if "cost_reconstruction_error" in results.columns:
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        plot_cost_reconstruction_error(results, ax=ax)
        fig.savefig(outdir / "paper_cost_reconstruction_error.png", bbox_inches="tight")
        plt.close(fig)


def create_slide_figures(results: pd.DataFrame, outdir: str | Path = "figures/slides") -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    set_slide_style()

    fig, _ = plot_slide_main_panel(results)
    fig.savefig(outdir / "slides_main_panel.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    plot_system_cost(results, ax=ax)
    fig.savefig(outdir / "slides_system_cost.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    plot_transmission_build(results, ax=ax)
    fig.savefig(outdir / "slides_transmission.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    plot_curtailment(results, ax=ax)
    fig.savefig(outdir / "slides_curtailment.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    plot_ldes_energy(results, ax=ax)
    fig.savefig(outdir / "slides_ldes_energy.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    plot_storage_comparison(results, ax=ax)
    fig.savefig(outdir / "slides_storage_comparison.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    plot_ldes_vs_transmission(results, ax=ax)
    fig.savefig(outdir / "slides_ldes_vs_transmission.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    plot_backup_capacity(results, ax=ax)
    fig.savefig(outdir / "slides_backup_capacity.png", bbox_inches="tight")
    plt.close(fig)