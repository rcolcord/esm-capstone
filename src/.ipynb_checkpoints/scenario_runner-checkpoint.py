import numpy as np
import pandas as pd
import pypsa

from typing import Any
from pathlib import Path

import network_builder

def summarize_network_results(
    n: pypsa.Network,
    scenario_id: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract a compact scenario summary from a solved PyPSA network.
    """
    row: dict[str, Any] = {"scenario_id": scenario_id, **metadata}
    row["objective"] = float(getattr(n, "objective", float("nan")))

    def add_grouped_capacity(
        table: pd.DataFrame,
        capacity_col: str = "p_nom_opt",
        energy_col: str | None = None,
    ) -> None:
        if (
            table.empty
            or capacity_col not in table.columns
            or "carrier" not in table.columns
        ):
            return

        for carrier, value in table.groupby("carrier", dropna=True)[capacity_col].sum().items():
            row[f"{carrier}_mw"] = float(value)

        if energy_col is not None and energy_col in table.columns:
            energy = (table[capacity_col] * table[energy_col]).groupby(
                table["carrier"], dropna=True
            ).sum()
            for carrier, value in energy.items():
                row[f"{carrier}_mwh"] = float(value)

    def add_grouped_capex(
        table: pd.DataFrame,
        total_key: str,
        capacity_col: str = "p_nom_opt",
    ) -> None:
        if (
            table.empty
            or capacity_col not in table.columns
            or "capital_cost" not in table.columns
            or "carrier" not in table.columns
        ):
            return

        capex = table[capacity_col] * table["capital_cost"]
        row[total_key] = float(capex.sum())

        for carrier, value in capex.groupby(table["carrier"], dropna=True).sum().items():
            row[f"{carrier}_capex"] = float(value)

    # -------------------------
    # Capacities
    # -------------------------
    if hasattr(n, "generators") and not n.generators.empty:
        add_grouped_capacity(n.generators, capacity_col="p_nom_opt")

    if hasattr(n, "storage_units") and not n.storage_units.empty:
        add_grouped_capacity(
            n.storage_units,
            capacity_col="p_nom_opt",
            energy_col="max_hours",
        )

    if hasattr(n, "links") and not n.links.empty:
        paid_links = n.links[n.links["capital_cost"] > 0]
        add_grouped_capacity(paid_links, capacity_col="p_nom_opt")

    # -------------------------
    # VRE curtailment
    # -------------------------
    if (
        hasattr(n, "generators")
        and hasattr(n, "generators_t")
        and hasattr(n.generators_t, "p")
        and hasattr(n.generators_t, "p_max_pu")
        and not n.generators.empty
        and "p_nom_opt" in n.generators.columns
    ):
        vre = n.generators[n.generators["carrier"].isin(["wind", "solar"])]

        vres_available = 0.0
        vres_dispatched = 0.0

        for gen_name, gen_row in vre.iterrows():
            if gen_name not in n.generators_t.p.columns:
                continue
            if gen_name not in n.generators_t.p_max_pu.columns:
                continue

            p_nom_opt = float(gen_row["p_nom_opt"])
            available = p_nom_opt * n.generators_t.p_max_pu[gen_name]
            dispatched = n.generators_t.p[gen_name]

            vres_available += float(available.sum())
            vres_dispatched += float(dispatched.sum())

        row["vres_available_mwh"] = vres_available
        row["vres_dispatched_mwh"] = vres_dispatched
        row["vres_curtailment_mwh"] = max(vres_available - vres_dispatched, 0.0)

    # -------------------------
    # System energy totals
    # -------------------------
    if hasattr(n, "loads_t") and hasattr(n.loads_t, "p"):
        total_load = float(n.loads_t.p.sum().sum())
        row["total_load_mwh"] = total_load
        if total_load > 0:
            row["system_cost_per_mwh"] = row["objective"] / total_load

    if hasattr(n, "generators_t") and hasattr(n.generators_t, "p"):
        row["total_generation_mwh"] = float(n.generators_t.p.sum().sum())

    # -------------------------
    # Cost components
    # -------------------------
    weights = (
        n.snapshot_weightings["objective"]
        if hasattr(n, "snapshot_weightings")
        and "objective" in n.snapshot_weightings.columns
        else pd.Series(1.0, index=n.snapshots)
    )

    if hasattr(n, "generators") and not n.generators.empty:
        add_grouped_capex(n.generators, total_key="generator_capex_total")

    if hasattr(n, "storage_units") and not n.storage_units.empty:
        add_grouped_capex(n.storage_units, total_key="storage_capex_total")

    if hasattr(n, "links") and not n.links.empty:
        paid_links = n.links[n.links["capital_cost"] > 0]
        add_grouped_capex(paid_links, total_key="transmission_capex_total")

    if (
        hasattr(n, "generators")
        and hasattr(n, "generators_t")
        and hasattr(n.generators_t, "p")
        and not n.generators.empty
        and "carrier" in n.generators.columns
    ):
        dispatch = n.generators_t.p.mul(weights, axis=0)

        gen_opex_total = 0.0
        for gen_name, gen_row in n.generators.iterrows():
            if gen_name not in dispatch.columns:
                continue

            marginal_cost = float(gen_row.get("marginal_cost", 0.0))
            cost = float((dispatch[gen_name] * marginal_cost).sum())
            gen_opex_total += cost

            carrier = gen_row.get("carrier")
            if carrier:
                row[f"{carrier}_opex"] = row.get(f"{carrier}_opex", 0.0) + cost

        row["generator_opex_total"] = gen_opex_total

    row["total_capex"] = (
        row.get("generator_capex_total", 0.0)
        + row.get("storage_capex_total", 0.0)
        + row.get("transmission_capex_total", 0.0)
    )
    row["total_opex"] = row.get("generator_opex_total", 0.0)
    row["reconstructed_total_cost"] = row["total_capex"] + row["total_opex"]
    row["cost_reconstruction_error"] = (
        row["objective"] - row["reconstructed_total_cost"]
    )

    return row


######################################################################################
######################################################################################

def run_scenario(
    scenario_id: str,
    snapshots: pd.DatetimeIndex,
    load: pd.DataFrame,
    wind_cf: pd.DataFrame,
    solar_cf: pd.DataFrame,
    *,
    co2_cap_tons: float,
    ldes_capex_per_mw: float,
    transmission_capex_per_mw: float,
    ldes_duration_hours: int,
    export_dir: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[pypsa.Network, dict[str, Any]]:
    """
    Build, solve, and summarize one scenario.
    """
    n = network_builder.build_network(
        snapshots=snapshots,
        load=load,
        wind_cf=wind_cf,
        solar_cf=solar_cf,
        co2_cap_tons=co2_cap_tons,
        ldes_capex_per_mw=ldes_capex_per_mw,
        transmission_capex_per_mw=transmission_capex_per_mw,
        ldes_duration_hours=ldes_duration_hours,
    )

    # for multi-hour resolution
    #RESOLUTION = 3 # 3 hours
    #n.set_snapshots(n.snapshots[::RESOLUTION])
    #n.snapshot_weightings.loc[:, :] = RESOLUTION

    status, condition = n.optimize(solver_name="highs")

    md = {} if metadata is None else metadata.copy()
    md["solver_status"] = status
    md["solver_condition"] = condition

    summary = summarize_network_results(n, scenario_id=scenario_id, metadata=md)

    if export_dir is not None:
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        # Summary row
        pd.DataFrame([summary]).to_csv(export_dir / f"{scenario_id}_summary.csv", index=False)

        # Optional detailed exports
        if hasattr(n, "generators") and not n.generators.empty:
            n.generators.to_csv(export_dir / f"{scenario_id}_generators.csv")
        if hasattr(n, "storage_units") and not n.storage_units.empty:
            n.storage_units.to_csv(export_dir / f"{scenario_id}_storage_units.csv")
        if hasattr(n, "links") and not n.links.empty:
            n.links.to_csv(export_dir / f"{scenario_id}_links.csv")

    return n, summary


######################################################################################

def run_core_scenarios(
    snapshots: pd.DatetimeIndex,
    load: pd.DataFrame,
    correlation_cases: dict[str, dict[str, pd.DataFrame]],
    *,
    ldes_duration_hours_list: list[int],
    ldes_capex_per_mw: float,
    transmission_capex_per_mw: float,
    co2_baseline_fraction: float = 0.05,
    export_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Run the core scenario matrix.

    Parameters
    ----------
    correlation_cases : dict
        Example:
        {
            "low":  {"wind_cf": wind_low_df,  "solar_cf": solar_low_df},
            "high": {"wind_cf": wind_high_df, "solar_cf": solar_high_df},
        }

    ldes_duration_hours_list : list[int]
        Example: [10, 24, 100]

    ldes_capex_per_mw : float
        Baseline annualized LDES capital cost input for build_network.

    transmission_capex_per_mw : float
        Baseline annualized transmission capital cost input for build_network.
    """

    total_load_mwh = load.sum().sum()
    baseline_emissions = total_load_mwh * 0.4 #tCO2 / MWh
    co2_cap_tons = baseline_emissions * co2_baseline_fraction

    rows: list[dict[str, Any]] = []

    for corr_case, cf_dict in correlation_cases.items():
        wind_cf = cf_dict["wind_cf"]
        solar_cf = cf_dict["solar_cf"]

        for duration in ldes_duration_hours_list:
            scenario_id = f"corr_{corr_case}_dur_{duration}h"

            metadata = {
                "renewable_correlation_case": corr_case,
                "ldes_duration_hours": duration,
                "ldes_capex_per_mw": ldes_capex_per_mw,
                "transmission_capex_per_mw": transmission_capex_per_mw,
                "co2_cap_tons": co2_cap_tons,
            }

            _, summary = run_scenario(
                scenario_id=scenario_id,
                snapshots=snapshots,
                load=load,
                wind_cf=wind_cf,
                solar_cf=solar_cf,
                co2_cap_tons=co2_cap_tons,
                ldes_capex_per_mw=ldes_capex_per_mw,
                transmission_capex_per_mw=transmission_capex_per_mw,
                ldes_duration_hours=duration,
                export_dir=Path(export_dir) / scenario_id if export_dir else None,
                metadata=metadata,
            )
            rows.append(summary)

    results = pd.DataFrame(rows).sort_values(
        ["renewable_correlation_case", "ldes_duration_hours"]
    )
    if export_dir is not None:
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(export_dir / "core_scenario_results.csv", index=False)

    return results.reset_index(drop=True)