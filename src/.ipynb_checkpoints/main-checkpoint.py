import numpy as np
import pandas as pd
import pypsa

import sys
from pathlib import Path

import network_builder
import util
import scenario_runner

# Initialize load and renewable profile data
load_data = pd.read_csv("data/clean_load.csv")
load_data["time_utc"] = pd.to_datetime(load_data["time_utc"], utc=True)
load_data = load_data.set_index("time_utc")

snapshots = load_data.index
nHours = len(snapshots)

load = pd.DataFrame(index=snapshots)
load["Region1"] = load_data["load_mw"] * .7
load["Region2"] = load_data["load_mw"] *  .5
load["Region3"] = load_data["load_mw"] * .3

wind_low, solar_low = util.load_trio_csv("data/low_correlation_trio_timeseries.csv")
wind_high, solar_high = util.load_trio_csv("data/high_correlation_trio_timeseries.csv")

# Sanity check
print(wind_low.index.equals(solar_low.index))
print(wind_high.index.equals(solar_high.index))
print(wind_low.index.tz)   # should be UTC

print(load.index.equals(wind_low.index))

# Make non-tz aware for PyPSA
snapshots = snapshots.tz_convert("UTC").tz_localize(None)
load.index = load.index.tz_convert("UTC").tz_localize(None)
wind_low.index = wind_low.index.tz_convert("UTC").tz_localize(None)
wind_high.index = wind_high.index.tz_convert("UTC").tz_localize(None)
solar_low.index = solar_low.index.tz_convert("UTC").tz_localize(None)
solar_high.index = solar_high.index.tz_convert("UTC").tz_localize(None)

correlation_cases = {
    "low": {
        "wind_cf": wind_low,
        "solar_cf": solar_low,
    },
    "high": {
        "wind_cf": wind_high,
        "solar_cf": solar_high,
    },
}

# Baseline annualized costs
ldes_capex_per_mw = 1_800_000
transmission_capex_per_mw = 400_000

# Run core scenario matrix and save results
core_results = scenario_runner.run_core_scenarios(
    snapshots=snapshots,
    load=load,
    correlation_cases=correlation_cases,
    ldes_duration_hours_list=[10, 24, 72, 100],
    ldes_capex_per_mw=ldes_capex_per_mw,
    transmission_capex_per_mw=transmission_capex_per_mw,
    co2_baseline_fraction=0.05,   # 95% reduction from gas-only baseline
    export_dir="results",
)

print(core_results)