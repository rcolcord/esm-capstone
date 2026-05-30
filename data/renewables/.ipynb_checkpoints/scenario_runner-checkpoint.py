import numpy as np
import pandas as pd
import pypsa

from src import network_builder

# rc todo : read

def summarize_network_results(
    n: pypsa.Network,
    scenario_id: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract a compact scenario summary from a solved PyPSA network.
    """
    row: dict[str, Any] = {"scenario_id": scenario_id, **metadata}

    # Objective
    row["objective"] = float(getattr(n, "objective", float("nan")))

    # Generator capacities
    if hasattr(n, "generators") and not n.generators.empty:
        gens = n.generators.copy()
        if "p_nom_opt" in gens.columns:
            for carrier in gens["carrier"].dropna().unique():
                mask = gens["carrier"] == carrier
                row[f"{carrier}_mw"] = float(gens.loc[mask, "p_nom_opt"].sum())

    # Storage capacities
    if hasattr(n, "storage_units") and not n.storage_units.empty:
        sus = n.storage_units.copy()
        if "p_nom_opt" in sus.columns:
            for carrier in sus["carrier"].dropna().unique():
                mask = sus["carrier"] == carrier
                row[f"{carrier}_mw"] = float(sus.loc[mask, "p_nom_opt"].sum())
                if "max_hours" in sus.columns:
                    row[f"{carrier}_mwh"] = float(
                        (sus.loc[mask, "p_nom_opt"] * sus.loc[mask, "max_hours"]).sum()
                    )

    # Transmission capacities
    if hasattr(n, "links") and not n.links.empty:
        links = n.links.copy()
        if "p_nom_opt" in links.columns:
            if "carrier" in links.columns:
                for carrier in links["carrier"].dropna().unique():
                    mask = links["carrier"] == carrier
                    row[f"{carrier}_mw"] = float(links.loc[mask, "p_nom_opt"].sum())
            else:
                row["links_mw"] = float(links["p_nom_opt"].sum())

    # Curtailment, if generator dispatch is available
    if (
        hasattr(n, "generators_t")
        and hasattr(n.generators_t, "p")
        and hasattr(n.generators_t, "p_max_pu")
        and hasattr(n, "generators")
        and not n.generators.empty
        and "p_nom_opt" in n.generators.columns
    ):
        vres_available = 0.0
        vres_dispatched = 0.0

        for gen_name, gen_row in n.generators.iterrows():
            if gen_row.get("carrier") not in {"wind", "solar"}:
                continue

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
    n = build_network(
        snapshots=snapshots,
        load=load,
        wind_cf=wind_cf,
        solar_cf=solar_cf,
        co2_cap_tons=co2_cap_tons,
        ldes_capex_per_mw=ldes_capex_per_mw,
        transmission_capex_per_mw=transmission_capex_per_mw,
        ldes_duration_hours=ldes_duration_hours,
    )

    #RESOLUTION = 1
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