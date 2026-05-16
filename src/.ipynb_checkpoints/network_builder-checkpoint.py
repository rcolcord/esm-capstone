import numpy as np
import pandas as pd
import pypsa

from util import annualized_capex

def build_network(
    snapshots: pd.DatetimeIndex,
    load: pd.DataFrame,
    wind_cf: pd.DataFrame,
    solar_cf: pd.DataFrame,
    co2_cap_tons: float,
    ldes_capex_per_mw: float,
    transmission_capex_per_mw: float,
    ldes_duration_hours: int = 24,
) -> pypsa.Network:

    n = pypsa.Network()
    n.set_snapshots(snapshots)

    n.snapshot_weightings.loc[:, "objective"] = 1.0
    n.snapshot_weightings.loc[:, "stores"] = 1.0
    n.snapshot_weightings.loc[:, "generators"] = 1.0

    required_regions = ["Region1", "Region2", "Region3"]

    if not load.index.equals(snapshots):
        raise ValueError("load index does not match snapshots")
    if not wind_cf.index.equals(snapshots):
        raise ValueError("wind_cf index does not match snapshots")
    if not solar_cf.index.equals(snapshots):
        raise ValueError("solar_cf index does not match snapshots")

    for df_name, df in [("load", load), ("wind_cf", wind_cf), ("solar_cf", solar_cf)]:
        missing = [r for r in required_regions if r not in df.columns]
        if missing:
            raise ValueError(f"{df_name} missing columns: {missing}")

    # Buses
    n.add("Carrier", "AC")
    buses = required_regions
    for bus in buses:
        n.add("Bus", bus, carrier="AC")

    # Load
    for region in load.columns:
        n.add("Load", f"load_{region}", bus=region, p_set=load[region])

    # Cost assumptions
    wacc = 0.07
    wind_capex_per_mw = 1_660_000
    solar_capex_per_mw = 1_480_000
    short_storage_capex_per_mw = 1_300_000
    backup_gas_capex_per_mw = 560_000

    wind_lifetime_years = 30
    solar_lifetime_years = 30
    short_storage_lifetime_years = 15
    ldes_lifetime_years = 20
    backup_gas_lifetime_years = 30
    transmission_lifetime_years = 40

    solar_fom_per_mw = 32000
    solar_fom_rate = solar_fom_per_mw / solar_capex_per_mw
    wind_fom_per_mw = 24000
    wind_fom_rate = wind_fom_per_mw / wind_capex_per_mw
    short_storage_fom_per_mw = 30000
    short_storage_fom_rate = short_storage_fom_per_mw / short_storage_capex_per_mw
    backup_gas_fom_per_mw = 15000
    backup_gas_fom_rate = backup_gas_fom_per_mw / backup_gas_capex_per_mw
    ldes_fom_rate = 0.02
    tx_fom_rate = 0.02

    wind_acc = annualized_capex(wind_capex_per_mw, wind_lifetime_years, wacc, wind_fom_rate)
    solar_acc = annualized_capex(solar_capex_per_mw, solar_lifetime_years, wacc, solar_fom_rate)
    short_storage_acc = annualized_capex(short_storage_capex_per_mw, short_storage_lifetime_years, wacc, short_storage_fom_rate)
    ldes_acc = annualized_capex(ldes_capex_per_mw, ldes_lifetime_years, wacc, ldes_fom_rate)
    backup_acc = annualized_capex(backup_gas_capex_per_mw, backup_gas_lifetime_years, wacc, backup_gas_fom_rate)
    tx_acc = annualized_capex(transmission_capex_per_mw, transmission_lifetime_years, wacc, tx_fom_rate)

    backup_marginal_cost = 45.0
    backup_emissions_t_per_mwh = 0.4

    # Generators
    for region in buses:
        n.add(
            "Generator",
            f"wind_{region}",
            bus=region,
            carrier="wind",
            p_nom_extendable=True,
            p_max_pu=wind_cf[region],
            capital_cost=wind_acc,
            marginal_cost=0.0,
        )

        n.add(
            "Generator",
            f"solar_{region}",
            bus=region,
            carrier="solar",
            p_nom_extendable=True,
            p_max_pu=solar_cf[region],
            capital_cost=solar_acc,
            marginal_cost=0.0,
        )

        n.add(
            "Generator",
            f"backup_{region}",
            bus=region,
            carrier="backup",
            p_nom_extendable=True,
            capital_cost=backup_acc,
            marginal_cost=backup_marginal_cost,
            efficiency=1.0,
        )

    # Storage
    for region in buses:
        n.add(
            "StorageUnit",
            f"battery4h_{region}",
            bus=region,
            carrier="battery4h",
            p_nom_extendable=True,
            max_hours=4.0,
            capital_cost=short_storage_acc,
            marginal_cost=0.0,
            efficiency_store=0.92,
            efficiency_dispatch=0.92,
            cyclic_state_of_charge=True,
        )

        n.add(
            "StorageUnit",
            f"ldes_{ldes_duration_hours}h_{region}",
            bus=region,
            carrier="ldes",
            p_nom_extendable=True,
            max_hours=float(ldes_duration_hours),
            capital_cost=ldes_acc,
            marginal_cost=0.0,
            efficiency_store=0.80,
            efficiency_dispatch=0.80,
            cyclic_state_of_charge=True,
        )

    # Transmission
    links = [
        ("Region1", "Region2"),
        ("Region1", "Region3"),
        ("Region2", "Region3"),
    ]

    for b0, b1 in links:
        n.add(
            "Link",
            f"tx_{b0}_{b1}",
            bus0=b0,
            bus1=b1,
            carrier="transmission",
            p_nom_extendable=True,
            p_min_pu=-1.0,
            p_max_pu=1.0,
            efficiency=0.97,
            marginal_cost=0.0,
            capital_cost=tx_acc,
        )

    # CO2 cap
    n.carriers.loc["wind", "co2_emissions"] = 0.0
    n.carriers.loc["solar", "co2_emissions"] = 0.0
    n.carriers.loc["battery4h", "co2_emissions"] = 0.0
    n.carriers.loc["ldes", "co2_emissions"] = 0.0
    n.carriers.loc["transmission", "co2_emissions"] = 0.0
    n.carriers.loc["backup", "co2_emissions"] = backup_emissions_t_per_mwh

    n.add(
        "GlobalConstraint",
        "co2_limit",
        type="primary_energy",
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=co2_cap_tons,
    )

    return n