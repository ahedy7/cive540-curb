"""
Delivery demand per block face, from the 2025 commercial premises.

Each occupied business gets daily deliveries under two scenarios. Rates are by sector
and live in analysis/delivery_rates.csv, one row per SCIAN (Canadian NAICS 2022) 3-digit code.

    low   Sanchez-Diaz (2017), Table 3: mean deliveries per week by sector for commercial
          establishments in Gothenburg, divided by 5 weekdays. One rate per business.
    high  Holguin-Veras et al. (2011), New York City models by SIC sector:
          deliveries/day = constant + rate x employees. Read from NCHRP Report 739,
          Table 24, which reports the same NYC models. The survey has no employee counts,
          so each row carries an assumed staff size. The employee terms are small.

Sectors with no NYC rate (services, schools) use the low rate in both scenarios.
Offices have no rate in either source: 0 in the low scenario, the "other services" rate
in the high one.

Units inside shopping centres (Eaton, Place Ville Marie etc.) are served by the centre's
own loading docks, so they are kept out of curb demand and counted separately.

Inputs:  data/derived/premises_2025.gpkg, data/derived/blockfaces.gpkg, analysis/delivery_rates.csv
Outputs: data/derived/demand.gpkg, layers "premises_demand" and "demand_by_face"

Usage:
    python analysis/demand.py
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
RATES = ROOT / "analysis" / "delivery_rates.csv"
OUT = DERIVED / "demand.gpkg"
SCENARIOS = ["low", "high"]


def main():
    faces = gpd.read_file(DERIVED / "blockfaces.gpkg")
    prem = gpd.read_file(DERIVED / "premises_2025.gpkg")
    rates = pd.read_csv(RATES, dtype={"scian3": str})
    rates["deliveries_low"] = rates.low_per_day
    rates["deliveries_high"] = rates.high_const + rates.high_per_emp * rates.assumed_fte

    p = prem[prem.occupied].copy()
    p["scian3"] = p.SCIAN.fillna("").astype(str).str[:3].where(lambda s: s.str.len() == 3, "000")
    p["in_mall"] = p.T_COMMERCE == "Centre commercial"
    missing = set(p.scian3) - set(rates.scian3)
    if missing:
        raise SystemExit(f"SCIAN codes with no row in {RATES.name}: {sorted(missing)}")
    p = p.merge(rates[["scian3", "label", "low_sector", "high_sic", "deliveries_low", "deliveries_high"]],
                on="scian3", how="left")

    cols = ["ID", "NOM_ETAB", "ADRESSE", "USAGE3", "scian3", "label", "low_sector", "high_sic",
            "in_mall", "blockface_id", "deliveries_low", "deliveries_high", "geometry"]
    p = gpd.GeoDataFrame(p[cols], geometry="geometry", crs=prem.crs)

    curb = p[~p.in_mall]
    by_face = curb.groupby("blockface_id").agg(premises=("ID", "size"),
                                               deliveries_low=("deliveries_low", "sum"),
                                               deliveries_high=("deliveries_high", "sum"))
    by_face["mall_premises"] = p[p.in_mall].groupby("blockface_id").size()
    out = faces[["blockface_id", "street", "side", "length_m", "street_class", "geometry"]].merge(
        by_face, left_on="blockface_id", right_index=True, how="left")
    out[["premises", "mall_premises"]] = out[["premises", "mall_premises"]].fillna(0).astype(int)
    for s in SCENARIOS:
        out[f"deliveries_{s}"] = out[f"deliveries_{s}"].fillna(0).round(2)
        out[f"per_100m_{s}"] = (out[f"deliveries_{s}"] / out.length_m * 100).round(2)

    p.to_file(OUT, layer="premises_demand", driver="GPKG")
    out.to_file(OUT, layer="demand_by_face", driver="GPKG")

    # report
    pd.set_option("display.width", 200)
    print(f"{len(p)} occupied premises: {len(curb)} on the curb, {p.in_mall.sum()} in shopping centres (docks)")
    print(curb.groupby("label").agg(n=("ID", "size"), low=("deliveries_low", "sum"), high=("deliveries_high", "sum"))
          .sort_values("high", ascending=False).round(1).to_string())
    print(f"\ndeliveries per weekday, curb premises: low {curb.deliveries_low.sum():.0f}, high {curb.deliveries_high.sum():.0f}")
    m = p[p.in_mall]
    print(f"mall premises (not on curb) would add: low {m.deliveries_low.sum():.0f}, high {m.deliveries_high.sum():.0f}")
    has = out[out.premises > 0]
    print(f"\nfaces with any curb premises: {len(has)} of {len(out)}")
    print(has[["deliveries_low", "deliveries_high"]].describe().round(2).to_string())
    print("\ntop 10 faces (high):")
    print(out.nlargest(10, "deliveries_high")[["blockface_id", "premises", "deliveries_low", "deliveries_high"]].to_string(index=False))
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
