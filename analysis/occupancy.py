"""
Curb occupancy from the side cameras, to replace two placeholders in loading.py:
    VACANT_SHARE  share of paid and free curb that is empty (drivers can pull in there)
    MISUSE_SHARE  share of loading curb taken by cars and taxis instead of commercial vehicles

Frames: side cameras only (L and R, one curb in view, about 15 m of it), kept on a sampled
face, moving (a stopped car films the same spot over and over). Each frame's spot on the face
(along_m) and clock time give the curb rule in force there and then, from the stretches built
by supply.py, strictest rule first. Only frames on paid, free, short-term or loading curb are
used. Elsewhere nobody may park, so occupancy says nothing about spare space.

A vision-language model (Claude Haiku 5.5) gets the frame and the frames 2 s before and after
(a first run on the frame alone counted cars driving past as parked) and answers: is the curb visible, what the
curb lane is used for, how much of it is taken (0, 1/4, 1/2, 3/4, full), and the vehicles
parked along it. Answers are cached in data/derived/vlm/occupancy_ctx/.

Shares are means over frames, so a long face counts more than a short one, as it should for a
share of curb length. Confidence intervals: bootstrap over faces (frames on one face are not
independent).

Outputs: data/derived/occupancy.csv (one row per frame), printed summary.

Usage:
    python analysis/occupancy.py --count   # frames and cost estimate, no API calls
    python analysis/occupancy.py           # ask the model (cached), then summarise
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from supply import USE_PRIORITY, active, in_season, parse_schedule   # noqa: E402
from vlm import CONTEXT_S, CONTEXT_SIDE, MODELS, image_block, load_key, neighbour   # noqa: E402

DERIVED = ROOT / "data" / "derived"
CACHE = DERIVED / "vlm" / "occupancy_ctx"   # v1 (NOW only) in occupancy/ counted moving cars
MODEL = "haiku"
WORKERS = 6
USES = ["paid", "free", "short_term", "loading"]   # curb where parking or loading is allowed
COMMERCIAL = {"delivery_van", "box_truck", "pickup"}
N_BOOT = 2000
DAYS = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]

SCHEMA = {
    "type": "object",
    "properties": {
        "curb_visible": {"type": "boolean"},
        "curb_lane": {"type": "string", "enum": ["parking", "travel_lane", "bike_lane", "construction", "other_use", "unclear"]},
        "occupied_share": {"type": "number", "enum": [0, 0.25, 0.5, 0.75, 1]},
        "vehicles": {"type": "array", "items": {"type": "string", "enum": [
            "car", "taxi", "delivery_van", "box_truck", "pickup", "bus", "motorcycle", "other"]}},
    },
    "required": ["curb_visible", "curb_lane", "occupied_share", "vehicles"],
    "additionalProperties": False,
}

PROMPT = """You get three photos from the same camera: BEFORE (2 seconds earlier), NOW and AFTER \
(2 seconds later). Describe only NOW. Use BEFORE and AFTER to see which vehicles move against the \
buildings and parked cars. If a frame is missing, judge from NOW alone.

NOW was taken from a car driving through downtown Montréal on a weekday morning. \
The camera points at the {side} curb of the street. Look only at the {side} curb lane, from the \
nearest part of it up to about 20 m ahead. Ignore the other side of the street and cross streets.

curb_visible: false if that curb lane can't be seen (blocked by a passing vehicle, blurred, dark).
curb_lane: what that lane is used for right now. parking (cars may stop along the curb, empty or not), \
travel_lane (cars drive in it), bike_lane, construction (fenced off or full of work equipment), \
other_use (a bike-share station, a terrace, planters or anything else built on the curb lane), unclear.
occupied_share: how much of that curb lane's length is taken by parked or stopped vehicles: \
0, 0.25, 0.5, 0.75 or 1. Count only vehicles standing still against the curb. A vehicle that \
moves between BEFORE and AFTER is traffic, not parked, even if it is close to the curb.
vehicles: each vehicle parked or stopped along that curb, once each. delivery_van = cargo van \
(Sprinter, Transit, ProMaster, courier vans), passenger minivans are car. box_truck = cube van or \
larger truck. taxi = taxi markings or roof light."""


# ---------------------------------------------------------------- frames and rules
def rules_by_face():
    s = gpd.read_file(DERIVED / "curb_supply.gpkg", layer="stretches", ignore_geometry=True)
    s = s[s.season.map(in_season)].copy()
    s["sched"] = s.schedule.map(parse_schedule)
    s["rank"] = s.use.map({u: i for i, u in enumerate(USE_PRIORITY)})
    return {k: g for k, g in s.groupby("blockface_id")}


def rule_at(stretches, along, day, minute):
    """Strictest use in force at this spot and time, 'free' if no rule covers it."""
    if stretches is None:
        return "free"
    here = stretches[(stretches.from_m <= along) & (stretches.to_m >= along)]
    here = here[[active(s, day, minute) for s in here.sched]]
    return "free" if here.empty else here.sort_values("rank").use.iat[0]


def frames():
    f = gpd.read_file(DERIVED / "frames.gpkg", ignore_geometry=True)
    sample = set(gpd.read_file(DERIVED / "sample.gpkg", ignore_geometry=True).blockface_id)
    f = f[f.side.isin(["L", "R"]) & f.keep & f.blockface_id.isin(sample) & ~f.stationary.astype(bool)].copy()
    t = pd.to_datetime(f.time_local)
    f["day"] = t.dt.dayofweek.map(dict(enumerate(DAYS)))
    f["minute"] = t.dt.hour * 60 + t.dt.minute
    rules = rules_by_face()
    f["use"] = [rule_at(rules.get(r.blockface_id), r.along_m, r.day, r.minute) for r in f.itertuples()]
    f = f[f.use.isin(USES) & np.array([(ROOT / p).exists() for p in f.path])]
    return f[["frame_id", "date", "side", "time_local", "blockface_id", "along_m", "use", "path"]].reset_index(drop=True)


# ---------------------------------------------------------------- model
def ask(client, path, side):
    model_id = MODELS[MODEL][0]
    content = []
    for label, p, px in [("BEFORE", neighbour(path, -CONTEXT_S), CONTEXT_SIDE), ("NOW", path, 1568),
                         ("AFTER", neighbour(path, CONTEXT_S), CONTEXT_SIDE)]:
        content += [{"type": "text", "text": label + ("" if p else ": missing")}] + ([image_block(p, px)] if p else [])
    content.append({"type": "text", "text": PROMPT.format(side=side)})
    r = client.messages.create(
        model=model_id,
        max_tokens=4000,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": content}],
    )
    if r.stop_reason != "end_turn":
        raise RuntimeError(f"stop_reason {r.stop_reason}")
    text = next(b.text for b in r.content if b.type == "text")
    return {"result": json.loads(text), "usage": {"input": r.usage.input_tokens, "output": r.usage.output_tokens}}


def run(f):
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = f[np.array([not (CACHE / f"{i}.json").exists() for i in f.frame_id])]
    print(f"{len(f)} frames, {len(todo)} to ask, {len(f) - len(todo)} cached")
    client = anthropic.Anthropic(api_key=load_key(), max_retries=6)
    failed = 0
    with ThreadPoolExecutor(WORKERS) as pool:
        side = {"L": "left", "R": "right"}
        jobs = {pool.submit(ask, client, ROOT / r.path, side[r.side]): r.frame_id for r in todo.itertuples()}
        for n, job in enumerate(as_completed(jobs), 1):
            try:
                (CACHE / f"{jobs[job]}.json").write_text(json.dumps(job.result(), indent=1))
            except Exception as e:
                failed += 1
                print(f"  {jobs[job]}: {e}")
            if n % 100 == 0:
                print(f"  {n} / {len(todo)}")
    _, p_in, p_out = MODELS[MODEL]
    used = [json.loads((CACHE / f"{i}.json").read_text())["usage"] for i in f.frame_id if (CACHE / f"{i}.json").exists()]
    cost = sum(u["input"] * p_in + u["output"] * p_out for u in used) / 1e6
    print(f"{len(used)} answers, ${cost:.2f} in total. {failed} failed")


# ---------------------------------------------------------------- summary
def boot(df, stat, seed=540):
    """Point estimate and 95% interval, resampling whole faces."""
    faces = df.blockface_id.unique()
    groups = {k: g for k, g in df.groupby("blockface_id")}
    rng = np.random.default_rng(seed)
    draws = [stat(pd.concat([groups[k] for k in rng.choice(faces, len(faces))])) for _ in range(N_BOOT)]
    return stat(df), *np.percentile(draws, [2.5, 97.5])


def summarise(f):
    rows = []
    for r in f.itertuples():
        p = CACHE / f"{r.frame_id}.json"
        if p.exists():
            a = json.loads(p.read_text())["result"]
            n = len(a["vehicles"])
            rows.append({**r._asdict(), "curb_visible": a["curb_visible"], "curb_lane": a["curb_lane"],
                         "occupied_share": a["occupied_share"], "vehicles": " ".join(a["vehicles"]),
                         "n_vehicles": n, "n_car_taxi": sum(v in ("car", "taxi") for v in a["vehicles"]),
                         "n_commercial": sum(v in COMMERCIAL for v in a["vehicles"])})
    d = pd.DataFrame(rows).drop(columns=["Index"])
    d.drop(columns="path").to_csv(DERIVED / "occupancy.csv", index=False)
    print(f"\n{len(d)} frames answered. Curb visible in {d.curb_visible.mean():.0%}. Lane read as:",
          d[d.curb_visible].curb_lane.value_counts().to_dict())
    ok = d[d.curb_visible & (d.curb_lane == "parking")]
    print(f"\nOccupancy where the lane is a parking lane ({len(ok)} frames, {ok.blockface_id.nunique()} faces):")
    for use, g in ok.groupby("use"):
        est, lo, hi = boot(g, lambda x: x.occupied_share.mean())
        print(f"  {use:11s} {len(g):4d} frames {g.blockface_id.nunique():3d} faces  occupied {est:.0%} (95% CI {lo:.0%} to {hi:.0%})")

    pf = ok[ok.use.isin(["paid", "free"])]
    est, lo, hi = boot(pf, lambda x: 1 - x.occupied_share.mean())
    print(f"\nVACANT_SHARE (paid + free) = {est:.2f}  (95% CI {lo:.2f} to {hi:.2f})")

    ld = ok[ok.use == "loading"]

    def misuse(x):   # occupied share times the cars' and taxis' part of the vehicles seen
        part = np.where(x.n_vehicles > 0, x.n_car_taxi / x.n_vehicles.clip(lower=1), 0)
        return float((x.occupied_share * part).mean())
    if len(ld):
        est, lo, hi = boot(ld, misuse)
        print(f"MISUSE_SHARE (loading)     = {est:.2f}  (95% CI {lo:.2f} to {hi:.2f}), from {len(ld)} frames on "
              f"{ld.blockface_id.nunique()} faces. Commercial vehicles hold "
              f"{float((ld.occupied_share * np.where(ld.n_vehicles > 0, ld.n_commercial / ld.n_vehicles.clip(lower=1), 0)).mean()):.2f}")
    print(f"-> {(DERIVED / 'occupancy.csv').relative_to(ROOT)}")


def main():
    f = frames()
    if "--count" in sys.argv:
        print(f"{len(f)} frames on {f.blockface_id.nunique()} faces:", f.use.value_counts().to_dict(),
              f"by day {f.date.value_counts().sort_index().to_dict()}")
        print(f"cost at about $0.001 a frame (three images, low effort): ${0.001 * len(f):.2f}")
        return
    run(f)
    summarise(f)


if __name__ == "__main__":
    main()
