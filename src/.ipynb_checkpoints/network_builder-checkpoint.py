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
    
    # Init network
    n = pypsa.Network()
    n.set_snapshots(snapshots)
    
    # Setup Regions as buses
    buses = {"reg1":  {"x": -120.0, "y": 36.0},
             "reg2": {"x": -122.0, "y": 46.0},
             "reg3": {"x": -107.0, "y": 43.0}}
    
    for bus, coords in buses.items():
        n.add("Bus", bus, x=coords["x"], y=coords["y"], carrier="AC")
        
    # Setup Load for each region
    for region in load.columns:
        n.add("Load", f"load_{region}", bus=region, p_set=load[region])
    
    # Cost assumptions # rc todo
    wacc = 0.07
    wind_capex_per_mw = 1_500_000
    solar_capex_per_mw = 900_000
    short_storage_capex_per_mw = 1_200_000
    backup_gas_capex_per_mw = 800_000
    
    # rc todo : lifetimes, O&M rates
    
    # Generation annualized capital costs [$/MW-year]
    wind_acc = annualized_capex(wind_capex_per_mw, 25, wacc, 0.03)
    solar_acc = annualized_capex(solar_capex_per_mw, 25, wacc, 0.02)
    
    # Storage annualized capital costs [$/MW-year]
    short_storage_acc = annualized_capex(short_storage_capex_per_mw, 15, wacc, 0.02)

    # LDES
    ldes_acc = annualized_capex(ldes_capex_per_mw, 20, wacc, 0.02)
    
    # Backup generation
    backup_acc = annualized_capex(backup_gas_capex_per_mw, 30, wacc, 0.03)
    backup_marginal_cost = 120.0  # $/MWh # rc todo
    backup_emissions_t_per_mwh = 0.4 # tCO2/MWh rc todo
    
    # Transmission annualized cost [$/MW-year]
    tx_acc = annualized_capex(transmission_capex_per_mw, 40, wacc, 0.02)
    
    # Setup Generators
    for region in buses:
        # Wind
        n.add(
            "Generator",
            f"wind_{region}",
            bus=region,
            carrier="wind",
            p_nom_extendable=True,
            p_max_pu=wind_cf[region],
            capital_cost=wind_acc,
            marginal_cost=0.0,
            #p_nom_max=20000,  # optional upper bound
        )
        
        # Solar
        n.add(
            "Generator",
            f"solar_{region}",
            bus=region,
            carrier="solar",
            p_nom_extendable=True,
            p_max_pu=solar_cf[region],
            capital_cost=solar_acc,
            marginal_cost=0.0,
            #p_nom_max=20000,
        )
        
        # Firm backup
        n.add(
            "Generator",
            f"backup_{region}",
            bus=region,
            carrier="backup",
            p_nom_extendable=True,
            capital_cost=backup_acc,
            marginal_cost=backup_marginal_cost,
            efficiency=1.0,
            #p_nom_max=20000,
        )
    
    # StorageUnit couples power and energy via max_hours.
    # Good for a first version of your model.
    for region in buses:
        # Short-duration battery (4h)
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
    
        # Long-duration storage
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
        ("reg1", "reg2"),
        ("reg1", "reg3"),
        ("reg2", "reg3"),
    ]
    
    for b0, b1 in links:
        n.add("Link", f"tx_{b0}_{b1}", bus0=b0, bus1=b1, p_nom_extendable=True,
            efficiency=1.0, marginal_cost=0.0, capital_cost=tx_capex
        )
    
        """
        # Optional reverse direction if you want symmetric explicit links.
        # For simple stylized systems, one bidirectional Link is often enough
        # if your PyPSA version uses signed dispatch; otherwise add reverse link.
        n.add(
            "Link",
            f"tx_{bus1}_{bus0}",
            bus0=bus1,
            bus1=bus0,
            carrier="transmission",
            p_nom_extendable=True,
            efficiency=1.0,
            marginal_cost=0.0,
            capital_cost=0.0,   # avoid double-counting build cost
            p_nom_min=0.0,
            p_nom_max=20000.0,
        )"""
    
    # CO2 Emissions cap
    
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