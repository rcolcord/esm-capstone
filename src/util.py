import itertools
import pandas as pd

def annuity(rate: float, years: int) -> float:
    """Simple annuity factor for converting CAPEX to annualized cost."""
    if rate == 0:
        return 1 / years
    return rate / (1 - (1 + rate) ** (-years))

def annualized_capex(capex_per_mw: float, lifetime_years: int, wacc: float, fixed_om_frac: float = 0.0) -> float:
    """
    Convert overnight CAPEX [$/MW] into annualized cost [$/MW-year].
    fixed_om_frac is fraction of CAPEX per year, e.g. 0.02 for 2%.
    """
    return capex_per_mw * (annuity(wacc, lifetime_years) + fixed_om_frac)

def trio_correlation_score(trio, profiles):
    """
    Compute average pairwise wind/solar correlation for a trio.
    """
    pairs = list(itertools.combinations(trio, 2))

    wind_corrs = []
    solar_corrs = []

    for a, b in pairs:
        df_a = profiles[a]
        df_b = profiles[b]

        wind_corrs.append(df_a["wind_cf"].corr(df_b["wind_cf"]))
        solar_corrs.append(df_a["solar_cf"].corr(df_b["solar_cf"]))

    avg_wind_corr = sum(wind_corrs) / len(wind_corrs)
    avg_solar_corr = sum(solar_corrs) / len(solar_corrs)
    avg_total_corr = (avg_wind_corr + avg_solar_corr) / 2

    return avg_total_corr

def coord_label(coord):
    """
    Turn a coordinate tuple into a string suitable for column names.
    Example: (35.5, -119.5) -> '35.5000_-119.5000'
    """
    lat, lon = coord
    return f"{lat:.4f}_{lon:.4f}"

def build_trio_timeseries(trio, profiles):
    """
    Given a trio of coordinates, return one DataFrame with columns:
    <coord>_wind_cf, <coord>_solar_cf, ...
    """
    dfs = []

    for coord in trio:
        df = profiles[coord].copy()
        label = coord_label(coord)

        df = df.rename(columns={
            "wind_cf": f"{label}_wind_cf",
            "solar_cf": f"{label}_solar_cf"
        })

        dfs.append(df)

    # Inner join on timestamp index
    combined = pd.concat(dfs, axis=1, join="inner").sort_index()
    return combined

def build_trio_timeseries(trio, profiles):
    """
    Given a trio of coordinates, return one DataFrame with columns:
    <coord>_wind_cf, <coord>_solar_cf, ...
    """
    dfs = []

    for coord in trio:
        df = profiles[coord].copy()
        label = coord_label(coord)

        df = df.rename(columns={
            "wind_cf": f"{label}_wind_cf",
            "solar_cf": f"{label}_solar_cf"
        })

        dfs.append(df)

    # Inner join on timestamp index
    combined = pd.concat(dfs, axis=1, join="inner").sort_index()
    return combined


def load_trio_csv(filepath):
    df = pd.read_csv(filepath)

    # --- Parse time ---
    df["time"] = pd.to_datetime(df["time_utc"], utc=True)
    df = df.set_index("time")

    # --- Split wind / solar ---
    wind_cols = [c for c in df.columns if "wind_cf" in c]
    solar_cols = [c for c in df.columns if "solar_cf" in c]

    wind_cf = df[wind_cols].copy()
    solar_cf = df[solar_cols].copy()

    # --- Optional: rename columns to Region1/2/3 ---
    # (helps PyPSA stay clean)
    wind_cf.columns = [f"Region{i+1}" for i in range(len(wind_cf.columns))]
    solar_cf.columns = [f"Region{i+1}" for i in range(len(solar_cf.columns))]

    return wind_cf, solar_cf