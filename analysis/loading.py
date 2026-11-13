"""
Loading model: how often a delivery truck finds no usable curb space, and how much loading
curb each block face needs.

For every block face and weekday hour (07:00 to 19:00):

    arrivals    daily deliveries (demand.py) x the hour's share of deliveries.
                Shares come from the Gothenburg cordon count in Sanchez-Diaz (2017), Table 9,
                delivery columns only (3PL, express courier and post, HoReCa, retail).
    usable curb curb a delivery driver can legally stop at, from the resolved supply rules
                (strictest rule wins, as in supply.py):
                    loading and short-term (P 15 min etc.) count in full
                    paid and free parking count at VACANT_SHARE, the share of it that is empty
                Spaces = usable metres / 10 m, kept as a fraction.
    P(no space) Erlang loss (Erlang B): trucks arrive at random, stay an exponential dwell time,
                and leave (double park) if every space is busy. Schmid et al. (2018) found dwell
                times close to exponential, which is what the formula needs. Fractional spaces
                are handled by interpolating between the two whole numbers.

Pooling: drivers cross the street, so the two faces of a street block share their trucks
and their spaces. P(no space) is computed per street block. The spaces the block needs are
split between its two faces by their share of the block's deliveries.

Four combinations: low or high demand (demand.py) x dwell 13.4 min (Kim et al. 2021, on-street
mean, Seattle) or 16.5 min (Schmid et al. 2018, median of legally parked trucks, NYC).

Sensitivity tests:
    face_alone   no pooling, each face on its own
    misuse       MISUSE_SHARE of loading curb is taken by other vehicles
    loading_only only signed loading curb counts (no short-term, no vacant parking)

Mismatch per block face and window, at the window's peak demand hour:
    required_m   10 m x the block's needed spaces (fewest that keep P(no space) under TARGET_P),
                 times the face's share of the block's deliveries
    supplied_m   usable curb at that hour (mean over the 10 weekday half-hours)
    mi           (required - supplied) / max(required, supplied), -1 oversupply to +1 undersupply.
                 0 when a face has neither.

Inputs:  data/derived/demand.gpkg, data/derived/curb_supply.gpkg (stretches), data/derived/sample.gpkg
Outputs: data/derived/loading.gpkg  layers "face_hour" (table) and "mismatch" (block face lines)
         data/derived/mismatch.csv  (the mismatch layer without geometry)
         data/derived/maps/mismatch_07-11.png

Usage:
    python analysis/loading.py
"""
import math
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import pyogrio

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from supply import USE_PRIORITY, WEEKDAYS, active, in_season, parse_schedule

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
OUT = DERIVED / "loading.gpkg"
OUT_CSV = DERIVED / "mismatch.csv"
MAPS = DERIVED / "maps"

SPACE_M = 10.0        # one delivery vehicle, Dalla Chiara et al. (2026): about 9.4 to 10 m
TARGET_P = 0.10       # design rule: under 10% of trucks find every space busy
VACANT_SHARE = 0.15   # PLACEHOLDER until video gives occupancy. Share of paid and free curb that is empty
MISUSE_SHARE = 0.30   # PLACEHOLDER until video gives it. Share of loading curb taken by other vehicles
DWELLS_MIN = {"13.4": 13.4, "16.5": 16.5}
DEMANDS = ["low", "high"]
HOURS = range(7, 19)
WINDOWS = {"07-11": (7, 11), "11-15": (11, 15), "15-19": (15, 19)}
GRID_ROTATION = 34 - 90   # same grid-north turn as route.py

# Sanchez-Diaz (2017) Table 9, % of all commercial vehicles entering per period.
# Columns: 3PL, express courier and post, HoReCa, retail. Construction, services and waste are left out.
CORDON = {
    (6.5, 7.5): (1.3, 0.3, 1.6, 0.7),
    (7.5, 8.5): (1.3, 1.6, 1.6, 0.0),
    (8.5, 9.5): (6.9, 1.3, 1.3, 1.6),
    (9.5, 10.5): (4.9, 0.3, 3.3, 3.6),
    (10.5, 11.5): (4.6, 0.7, 1.6, 1.0),
    (11.5, 12.5): (2.6, 0.7, 0.3, 0.7),
    (12.5, 13.5): (4.9, 1.3, 1.6, 0.7),
    (13.5, 14.5): (1.6, 0.7, 1.0, 0.7),
    (14.5, 15.5): (1.0, 0.3, 0.7, 0.3),
    (15.5, 16.5): (2.0, 0.7, 0.0, 1.0),
    (16.5, 18.0): (0.0, 0.3, 0.3, 0.3),
}


def hourly_shares():
    """Share of a day's deliveries arriving in each clock hour 07 to 18.
    Cordon periods start on the half hour, so each period is spread evenly over its half-hours.
    The 06:30 to 07:00 half-hour falls outside our day, so the shares sum to a bit under 1."""
    half = {}
    for (a, b), cols in CORDON.items():
        n = round((b - a) * 2)
        for k in range(n):
            half[a + k / 2] = sum(cols) / n
    total = sum(half.values())
    return {h: (half.get(h, 0) + half.get(h + 0.5, 0)) / total for h in HOURS}


def erlang_b(c, a):
    """Probability that an arriving truck finds all c spaces busy, offered load a (Erlangs).
    Non-integer c is interpolated between floor(c) and floor(c) + 1."""
    if a <= 0:
        return 0.0
    n = math.floor(c)
    b = 1.0
    for k in range(1, n + 1):
        b = a * b / (k + a * b)
    nxt = a * b / (n + 1 + a * b)
    return b + (c - n) * (nxt - b)


def spaces_needed(a, target=TARGET_P):
    if a <= 0:
        return 0
    c = 1
    while erlang_b(c, a) >= target:
        c += 1
    return c


def curb_by_slot(faces, stretches):
    """Metres of loading, short-term and parking (paid + free) per face, weekday and half-hour,
    as a DataFrame indexed by (blockface_id, day, minute)."""
    rank = {u: i for i, u in enumerate(USE_PRIORITY)}
    st = stretches[stretches.season.map(in_season)].copy()
    st["sched"] = st.schedule.map(parse_schedule)
    slots = [(d, m + 15) for d in WEEKDAYS for m in range(HOURS[0] * 60, (HOURS[-1] + 1) * 60, 30)]
    rows = []
    for f in faces.itertuples():
        L = f.length_m
        mine = st[st.blockface_id == f.blockface_id]
        cuts = sorted({0.0, L, *mine.from_m.clip(0, L), *mine.to_m.clip(0, L)})
        pieces = [(b - a, mine[(mine.from_m <= a + 0.01) & (mine.to_m >= b - 0.01)])
                  for a, b in zip(cuts, cuts[1:]) if b - a >= 0.01]
        for day, minute in slots:
            m = {"loading": 0.0, "short_term": 0.0, "parking": 0.0}
            for length, here in pieces:
                uses = [u for u, sc in zip(here.use, here.sched) if active(sc, day, minute)]
                use = min(uses, key=rank.get) if uses else "free"
                key = "parking" if use in ("paid", "free") else use
                if key in m:
                    m[key] += length
            rows.append({"blockface_id": f.blockface_id, "day": day, "minute": minute, **m})
    return pd.DataFrame(rows).set_index(["blockface_id", "day", "minute"])


def face_hour_table(demand, curb, shares):
    """One row per face, hour, demand scenario and dwell. Curb and P are averaged over the
    10 weekday half-hours in the hour, since signs can change at the half hour or by day."""
    curb = curb.assign(
        usable=curb.loading + curb.short_term + VACANT_SHARE * curb.parking,
        misuse=(1 - MISUSE_SHARE) * curb.loading + curb.short_term + VACANT_SHARE * curb.parking)
    d = demand.set_index("blockface_id")
    block = d.index.to_series().str.rsplit("_", n=1).str[0]
    rows = []
    for h in HOURS:
        slots = [(day, h * 60 + m) for day in WEEKDAYS for m in (15, 45)]
        c = curb[curb.index.get_level_values("minute").isin([s[1] for s in slots])]
        # spaces per face and slot, as arrays over the 10 slots
        sp = {col: (c[col] / SPACE_M).unstack(["day", "minute"]).reindex(d.index).fillna(0)
              for col in ("usable", "misuse", "loading")}
        bsp = {col: v.groupby(block).sum() for col, v in sp.items()}
        for s in DEMANDS:
            lam = d[f"deliveries_{s}"] * shares[h]
            blam = lam.groupby(block).sum()
            for dname, dwell in DWELLS_MIN.items():
                a, ba = lam * dwell / 60, blam * dwell / 60
                for fid in d.index:
                    blk = block[fid]
                    p = lambda col, load, spaces: float(np.mean([erlang_b(k, load) for k in spaces]))
                    need = spaces_needed(ba[blk])
                    share = lam[fid] / blam[blk] if blam[blk] > 0 else 0.0
                    rows.append({
                        "blockface_id": fid, "block": blk, "hour": h, "demand": s, "dwell_min": dname,
                        "arrivals": lam[fid], "load_erl": a[fid],
                        "usable_m": float(sp["usable"].loc[fid].mean() * SPACE_M),
                        "loading_m": float(sp["loading"].loc[fid].mean() * SPACE_M),
                        "p_no_space": p("usable", ba[blk], bsp["usable"].loc[blk]),
                        "p_face_alone": p("usable", a[fid], sp["usable"].loc[fid]),
                        "p_misuse": p("misuse", ba[blk], bsp["misuse"].loc[blk]),
                        "p_loading_only": p("loading", ba[blk], bsp["loading"].loc[blk]),
                        "block_spaces_needed": need,
                        "required_m": need * SPACE_M * share,
                    })
    t = pd.DataFrame(rows)
    t["missed"] = t.arrivals * t.p_no_space   # trucks per hour that find no space
    return t


def mismatch_table(fh, demand, shares):
    rows = []
    for w, (h0, h1) in WINDOWS.items():
        peak = max(range(h0, h1), key=shares.get)
        win = fh[(fh.hour >= h0) & (fh.hour < h1)]
        missed = win.groupby(["blockface_id", "demand", "dwell_min"]).missed.sum().rename("missed_in_window")
        at_peak = fh[fh.hour == peak].set_index(["blockface_id", "demand", "dwell_min"])
        t = at_peak.join(missed).reset_index()
        t["window"], t["peak_hour"] = w, peak
        rows.append(t)
    m = pd.concat(rows, ignore_index=True)
    m["supplied_m"] = m.usable_m
    m["shortfall_m"] = m.required_m - m.supplied_m
    big = np.maximum(m.required_m, m.supplied_m)
    m["mi"] = np.where(big > 0, m.shortfall_m / big.where(big > 0, 1), 0.0)
    m = m.merge(demand[["blockface_id", "deliveries_low", "deliveries_high"]], on="blockface_id")
    m["deliveries_day"] = np.where(m.demand == "low", m.deliveries_low, m.deliveries_high)
    keep = ["blockface_id", "block", "window", "demand", "dwell_min", "peak_hour", "deliveries_day",
            "arrivals", "block_spaces_needed", "required_m", "supplied_m", "loading_m", "shortfall_m", "mi",
            "p_no_space", "p_face_alone", "p_misuse", "p_loading_only", "missed_in_window"]
    return m[keep].round(3)


def draw(faces, m, path):
    """2 x 2 maps of the mismatch index in the 07-11 window, one per combination."""
    origin = faces.union_all().centroid
    g = faces.copy()
    g["geometry"] = g.geometry.rotate(GRID_ROTATION, origin=origin)
    fig, axes = plt.subplots(2, 2, figsize=(20, 9.5), gridspec_kw={"hspace": 0.12, "wspace": 0.03})
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    for ax, (s, d) in zip(axes.flat, [(s, d) for s in DEMANDS for d in DWELLS_MIN]):
        sub = g.merge(m[(m.window == "07-11") & (m.demand == s) & (m.dwell_min == d)], on="blockface_id")
        empty = (sub.required_m == 0) & (sub.supplied_m == 0)
        sub[empty].plot(ax=ax, color="#d0d0d0", lw=1)
        sub[~empty].plot(ax=ax, column="mi", cmap="RdBu_r", norm=norm, lw=2.4)
        short = sub.shortfall_m.clip(lower=0).sum()
        ax.set_title(f"{s} demand, {d} min dwell:  {(sub.mi > 0.25).sum()} faces clearly short (MI > 0.25), "
                     f"{short / 1000:.1f} km short in total", fontsize=12)
        ax.set_axis_off(); ax.set_aspect("equal")
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm), ax=axes, orientation="horizontal",
                      shrink=0.45, pad=0.03, aspect=40)
    cb.set_label("Mismatch index:  blue = more usable curb than needed, red = not enough.  Grey = no demand and no usable curb.")
    fig.suptitle("Delivery curb mismatch, weekdays 07:00 to 11:00, peak hour 09:00 to 10:00 (map turned to grid north)",
                 fontsize=15, y=0.95)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    demand = gpd.read_file(DERIVED / "demand.gpkg", layer="demand_by_face")
    stretches = gpd.read_file(DERIVED / "curb_supply.gpkg", layer="stretches")
    sample = set(gpd.read_file(DERIVED / "sample.gpkg").blockface_id)
    shares = hourly_shares()

    curb = curb_by_slot(demand, stretches)
    fh = face_hour_table(demand, curb, shares)
    m = mismatch_table(fh, demand, shares)

    lines = demand[["blockface_id", "street", "side", "length_m", "street_class", "geometry"]].copy()
    lines["in_sample"] = lines.blockface_id.isin(sample)
    out = lines.merge(m, on="blockface_id")
    OUT.unlink(missing_ok=True)
    pyogrio.write_dataframe(fh.round(4), OUT, layer="face_hour", driver="GPKG")   # plain table, no geometry
    out.to_file(OUT, layer="mismatch", driver="GPKG")
    out.drop(columns="geometry").to_csv(OUT_CSV, index=False)
    draw(lines, m, MAPS / "mismatch_07-11.png")

    # report
    pd.set_option("display.width", 200)
    print("hourly share of deliveries:", {h: round(v, 3) for h, v in shares.items()},
          f"(07-19 total {sum(shares.values()):.3f})")
    for w in WINDOWS:
        x = m[m.window == w]
        print(f"\nwindow {w}, peak hour {x.peak_hour.iloc[0]}:00")
        print(x.groupby(["demand", "dwell_min"]).apply(lambda t: pd.Series({
            "short_mi>0.25": (t.mi > 0.25).sum(),
            "over_mi<-0.25": (t.mi < -0.25).sum(),
            "required_km": t.required_m.sum() / 1000,
            "usable_km": t.supplied_m.sum() / 1000,
            "short_km": t.shortfall_m.clip(lower=0).sum() / 1000,
            "trucks_no_space": t.missed_in_window.sum(),
        }), include_groups=False).round(2).to_string())
    print("\nshare of the day's trucks finding no space (07-19), by combination and test:")
    tests = ["p_no_space", "p_face_alone", "p_misuse", "p_loading_only"]
    print(fh.groupby(["demand", "dwell_min"]).apply(
        lambda t: pd.Series({c: (t.arrivals * t[c]).sum() / t.arrivals.sum() for c in tests}),
        include_groups=False).round(3).to_string())
    print("\ntop 12 faces short of curb, 07-11, high/16.5:")
    t = m[(m.window == "07-11") & (m.demand == "high") & (m.dwell_min == "16.5")]
    print(t.nlargest(12, "shortfall_m")[["blockface_id", "deliveries_day", "required_m", "supplied_m",
                                         "loading_m", "shortfall_m", "p_no_space"]].round(2).to_string(index=False))
    print(f"\nMI distribution 07-11 high/16.5:\n{pd.cut(t.mi, [-1.01, -0.5, -0.1, 0.1, 0.5, 1]).value_counts().sort_index().to_string()}")
    print(f"-> {OUT.relative_to(ROOT)}, {OUT_CSV.relative_to(ROOT)}, {MAPS.relative_to(ROOT)}/mismatch_07-11.png")


if __name__ == "__main__":
    main()
