"""
Observed double parking per street block, and the test of the loading model: do blocks the
model calls short of loading space have more double parking?

Events. The partner reviewed every frame the VLM flagged (validation/review_partner_*.json,
the latest file wins per frame). Coding rule (Nov 18):
    - in the travel lane, fully or half on the sidewalk: double parked (half = partial)
    - on the sidewalk only: not double parking, kept as "sidewalk"
    - a bus at its own stop: not double parking
Positive frames of one day are one event when they fall on the same street block within
GAP_S seconds, across cameras (on Wednesday the F and L phones filmed at the same time).

Unit = street block (both faces). On one-way streets the forward camera can't tell which curb a
truck is on, and the loading model already pools both faces of a block.

Exposure = 100 m of curb filmed: sum over days and faces of share_seen x face length / 100,
from data/derived/coverage.csv. A block filmed on three days counts three times.

Model: Poisson GLM, events ~ predictor, offset log(exposure), robust (HC1) errors. With about
ten events a negative binomial can't estimate its extra dispersion well, so it is a check.
Predictors from loading.py, 07-11 window (filming was 08:15 to 11:00): the block's mismatch
index and P(no space) at the peak hour, and demand alone (deliveries a day, log2, so the rate
ratio is per doubling). Main scenario: high demand, 13.4 min dwell.

Outputs: data/derived/dp_events.csv, dp_blocks.csv, printed results.

Usage:
    python analysis/double_parking.py
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools import add_constant   # statsmodels.api fails to import with this pandas

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
VALID = ROOT / "validation"
GAP_S = 60                    # same block, under a minute apart = same vehicle(s)
SIDEWALK_ONLY = {"20261113_L_4_000225", "20261113_L_4_000226", "20261113_L_4_000227"}
PARTIAL = [("20261111_F_1", 688, 692), ("20261111_F_1", 1039, 1041), ("20261111_F_1", 1638, 1643)]
MAIN = ("high", 13.4)
WINDOW = "07-11"
SCALE = {"mi": 10, "p_no_space": 10, "log2_deliveries": 1}   # rate ratio per +0.1, per +0.1, per doubling


# ---------------------------------------------------------------- events
def reviews():
    out = {}
    for p in sorted(VALID.glob("review_partner_*.json")):   # dated names sort in time order
        out.update(json.loads(p.read_text())["reviews"])
    r = pd.DataFrame.from_dict(out, orient="index").rename_axis("frame_id").reset_index()
    lst = pd.read_csv(VALID / "review_list.csv")
    missing = set(lst.frame_id) - set(r.frame_id)
    print(f"{len(r)} frames reviewed, {len(missing)} on the list without an answer")
    return lst.drop(columns="cls").merge(r, on="frame_id")


def code(d):
    """Apply the coding rule: 'double', 'sidewalk' or 'no' per frame."""
    bus_at_stop = (d.answer == "in_bus_stop") & (d.cls == "bus")
    out = np.where((d.answer == "no") | bus_at_stop, "no", "double")
    out = np.where(d.frame_id.isin(SIDEWALK_ONLY), "sidewalk", out)
    return out


def is_partial(frame_id):
    clip, n = frame_id.rsplit("_", 1)
    return any(clip == c and a <= int(n) <= b for c, a, b in PARTIAL)


def events(d, block_of):
    d = d.assign(code=code(d), block=d.blockface_id.map(block_of), time=pd.to_datetime(d.time))
    p = d[d.code != "no"].sort_values("time")
    rows, cur = [], None
    for r in p.itertuples():
        if cur and r.block == cur["block"] and r.code == cur["code"] and (r.time - cur["end"]).total_seconds() <= GAP_S:
            cur["end"] = r.time
            cur["frames"].append(r.frame_id)
            cur["cameras"].add(r.camera)
            cur["vehicles"] = max(cur["vehicles"], r.n)
            cur["classes"].add(r.cls)
            continue
        cur = {"block": r.block, "code": r.code, "date": r.time.date(), "start": r.time, "end": r.time,
               "frames": [r.frame_id], "cameras": {r.camera}, "vehicles": r.n, "classes": {r.cls}, "note": r.note}
        rows.append(cur)
    e = pd.DataFrame(rows)
    e["partial"] = e.frames.map(lambda f: any(is_partial(x) for x in f))
    e["n_frames"] = e.frames.map(len)
    for c in ["cameras", "classes"]:
        e[c] = e[c].map(lambda s: " ".join(sorted(s)))
    e["frames"] = e.frames.map(lambda f: f"{f[0]} .. {f[-1]}")
    e.insert(0, "event", range(1, len(e) + 1))
    return e


# ---------------------------------------------------------------- blocks
def blocks():
    s = gpd.read_file(DERIVED / "sample.gpkg", ignore_geometry=True)
    cov = pd.read_csv(DERIVED / "coverage.csv").merge(s[["blockface_id", "block", "length_m"]], on="blockface_id")
    cov["curb_100m"] = cov.share_seen * cov.length_m / 100
    b = cov.groupby("block").agg(exposure_100m=("curb_100m", "sum"), days=("date", "nunique"))
    b = b.join(s.groupby("block").agg(stratum=("stratum", "first"), street_class=("street_class", "first"),
                                      length_m=("length_m", "sum"), premises=("premises", "sum")))
    return b, s.set_index("blockface_id").block


def predictors(demand, dwell):
    m = pd.read_csv(DERIVED / "mismatch.csv")
    m = m[(m.window == WINDOW) & (m.demand == demand) & (m.dwell_min == dwell)]
    g = m.groupby("block").agg(required_m=("required_m", "sum"), supplied_m=("supplied_m", "sum"),
                               p_no_space=("p_no_space", "first"), deliveries_day=("deliveries_day", "sum"))
    top = np.maximum(g.required_m, g.supplied_m)
    g["log2_deliveries"] = np.log2(1 + g.deliveries_day)
    g["mi"] = np.where(top > 0, (g.required_m - g.supplied_m) / top.where(top > 0, 1), 0.0)
    return g


def fit(b, x):
    """Poisson rate ratio per SCALE step of x with HC1 errors, and the NB rate ratio as a check."""
    X = add_constant(b[[x]] * SCALE[x])
    off = np.log(b.exposure_100m)
    po = GLM(b.events, X, family=families.Poisson(), offset=off).fit(cov_type="HC1")
    try:
        nb = NegativeBinomial(b.events, X, offset=off).fit(disp=False)
        nb_rr = np.exp(nb.params[x])
    except Exception:
        nb_rr = np.nan
    lo, hi = np.exp(po.conf_int().loc[x])
    return {"rr": np.exp(po.params[x]), "ci_lo": lo, "ci_hi": hi, "p": po.pvalues[x], "nb_rr": nb_rr}


def main():
    b, block_of = blocks()
    e = events(reviews(), block_of)
    e.to_csv(DERIVED / "dp_events.csv", index=False)
    dp = e[e.code == "double"]
    print(f"\n{len(dp)} double-parking events on {dp.block.nunique()} blocks ({dp.partial.sum()} partial, "
          f"{dp.vehicles.sum()} vehicles), {(e.code == 'sidewalk').sum()} sidewalk-only stop(s):")
    print(e[["event", "date", "block", "code", "partial", "cameras", "vehicles", "classes", "n_frames"]].to_string(index=False))

    b = b[b.exposure_100m > 0].copy()
    b["events"] = dp.block.value_counts().reindex(b.index, fill_value=0)
    b["events_full"] = dp[~dp.partial].block.value_counts().reindex(b.index, fill_value=0)
    b = b.join(predictors(*MAIN))
    b["rate_per_km"] = b.events / b.exposure_100m * 10
    b.round(3).to_csv(DERIVED / "dp_blocks.csv")

    print(f"\n{len(b)} blocks filmed, {b.exposure_100m.sum() * 100 / 1000:.1f} km of curb filmed over all days, "
          f"{b.events.sum()} events = {b.events.sum() / b.exposure_100m.sum() * 10:.2f} per km of curb passed")
    short = b.mi > 0
    for name, g in [("short of space (MI > 0)", b[short]), ("enough space (MI <= 0)", b[~short])]:
        print(f"  {name:24s} {len(g):2d} blocks, {g.events.sum():2d} events, "
              f"{g.events.sum() / g.exposure_100m.sum() * 10:.2f} per km")
    busy = b.deliveries_day >= 9
    for name, g in [("9+ deliveries a day", b[busy]), ("under 9 a day", b[~busy])]:
        print(f"  {name:24s} {len(g):2d} blocks, {g.events.sum():2d} events, "
              f"{g.events.sum() / g.exposure_100m.sum() * 10:.2f} per km")

    rows = []
    for demand in ["high", "low"]:
        for dwell in [13.4, 16.5]:
            bb = b.drop(columns=list(predictors(*MAIN).columns)).join(predictors(demand, dwell))
            for x in SCALE:
                for outcome in ["events", "events_full"]:
                    r = fit(bb.assign(events=bb[outcome]), x)
                    rho = spearmanr(bb[x], bb[outcome] / bb.exposure_100m).correlation
                    rows.append({"demand": demand, "dwell": dwell, "x": x, "outcome": outcome, **r, "spearman": rho})
    res = pd.DataFrame(rows)
    print("\nRate ratio of double parking per +0.1 in MI or P(no space), per doubling of deliveries (Poisson, robust 95% CI):")
    print(res.round(3).to_string(index=False))
    res.round(4).to_csv(DERIVED / "dp_model.csv", index=False)
    print("-> data/derived/dp_events.csv, dp_blocks.csv, dp_model.csv")


if __name__ == "__main__":
    main()
