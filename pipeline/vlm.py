"""
Step 7, VLM extraction: ask a vision-language model what is at the curb in each frame.

The questions match the hand-labelling page (validation/label.html) one for one, so the
model's answers can be scored straight against the labels:
    status            ok, partial (curb partly hidden) or unusable
    signs             sign types in view: no_stopping, no_parking, loading, paid, reserved, other
    sign_text         French text of the signs, as read
    features          hydrant, driveway, bus_stop, bike_lane, yellow_paint
    vehicles          every stopped vehicle: class and position (parked_curb, double_parked,
                      in_bike_lane, in_bus_stop) and whether it is being loaded
    loading_activity  goods being moved anywhere in the frame
Moving traffic is left out. The answer is forced into this shape with a JSON schema.

Every answer is cached on disk, one file per model and frame, so nothing is paid for twice.
Token use is saved with it, and the run prints its real cost.

Frame sets:
    test    the first 100 frames of the validation sample (the ones labelled first)
    send    frames flagged send by detect.py

Usage (the API key is read from .env):
    python pipeline/vlm.py sonnet test          # 100 test frames on Sonnet
    python pipeline/vlm.py haiku test 10        # first 10 only
    python pipeline/vlm.py sonnet send          # full run
"""
import base64
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
import geopandas as gpd
import pandas as pd
from PIL import Image

from common import DERIVED, ROOT

MODELS = {   # id, $ per million input and output tokens
    "haiku": ("claude-haiku-5-5", 0.10, 0.50),
    "sonnet": ("claude-sonnet-5-5", 2.00, 10.00),
    "opus": ("claude-opus-5-5", 4.00, 20.00),
}
EFFORT = "medium"
MAX_SIDE = 1568      # longer side in pixels. Larger images are shrunk by the API anyway
WORKERS = 6
N_TEST = 100
OUT = DERIVED / "vlm"

SIGNS = ["no_stopping", "no_parking", "loading", "paid", "reserved", "other"]
FEATURES = ["hydrant", "driveway", "bus_stop", "bike_lane", "yellow_paint"]
CLASSES = ["car", "taxi", "delivery_van", "box_truck", "pickup", "cargo_bike", "bus"]
POSITIONS = ["parked_curb", "double_parked", "in_bike_lane", "in_bus_stop"]

SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["ok", "partial", "unusable"]},
        "signs": {"type": "array", "items": {"type": "string", "enum": SIGNS}},
        "sign_text": {"type": "string"},
        "features": {"type": "array", "items": {"type": "string", "enum": FEATURES}},
        "vehicles": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "class": {"type": "string", "enum": CLASSES},
                "position": {"type": "string", "enum": POSITIONS},
                "loading": {"type": "boolean"},
            },
            "required": ["class", "position", "loading"],
            "additionalProperties": False,
        }},
        "loading_activity": {"type": "boolean"},
    },
    "required": ["status", "signs", "sign_text", "features", "vehicles", "loading_activity"],
    "additionalProperties": False,
}

PROMPT = """This photo was taken from a car driving through downtown Montréal on a weekday morning. \
Describe what is at the curb of the street the car is on, both curbs if you can see them. \
Ignore cross streets in the distance.

status: ok if the curb is clearly visible. partial if it is partly hidden or far. \
unusable if it is too blurry, too dark or not visible (then leave the lists empty).

signs: the types of parking signs you can read, each type once.
- "Arrêt interdit" = no_stopping
- "Stationnement interdit" (P with a red circle and slash) = no_parking
- "Livraison seulement", "Débarcadère", "Zone de livraison" = loading
- payment sign, "Payez ici", or a parking meter = paid
- "Réservé", permit zones (S3R, "excepté détenteurs de permis"), taxi stand, hotel, disabled = reserved
- time limits ("P 60 min") and any other parking sign = other
Ignore traffic signs (stop, one way, speed limit, turn rules).
sign_text: the sign text as you read it, in French, signs separated by " | ". Empty if none.

features: hydrant, driveway (a curb cut into a lot or garage), bus_stop (bus stop sign or shelter), \
bike_lane (painted or protected), yellow_paint (yellow curb paint).

vehicles: every STOPPED vehicle along the curbs. Leave out moving traffic and cars waiting at a light.
- parked_curb: stopped against the curb, legal or not
- double_parked: stopped in a traffic lane beside the curb lane or beside parked cars, not waiting in traffic
- in_bike_lane: stopped on or across a bike lane
- in_bus_stop: stopped at the curb in a bus stop
Count each vehicle once, in one position. \
delivery_van = cargo van (Sprinter, Transit, ProMaster, courier vans). Passenger minivans are car. \
box_truck = cube van or larger truck. taxi = taxi markings or roof light.
loading: true if goods are being moved at that vehicle (a person carrying goods, a hand truck, \
open cargo doors with goods).

loading_activity: true if goods are being moved anywhere in the photo."""


def load_key():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("ANTHROPIC_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("No ANTHROPIC_API_KEY in .env")


def frame_set(name):
    """(frame_id, image path) pairs."""
    if name == "test":
        s = pd.read_csv(ROOT / "validation" / "sample.csv").head(N_TEST)
        return [(r.frame_id, ROOT / "validation" / "frames" / f"{r.frame_id}.jpg") for r in s.itertuples()]
    d = pd.read_csv(DERIVED / "detect" / "frames.csv")
    where = gpd.read_file(DERIVED / "frames.gpkg", ignore_geometry=True).set_index("frame_id").path
    return [(i, ROOT / where[i]) for i in d[d.send].frame_id]


def image_b64(path):
    im = Image.open(path).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return base64.standard_b64encode(buf.getvalue()).decode()


def ask(client, model_id, path):
    r = client.messages.create(
        model=model_id,
        max_tokens=8000,
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64(path)}},
            {"type": "text", "text": PROMPT},
        ]}],
    )
    if r.stop_reason != "end_turn":
        raise RuntimeError(f"stop_reason {r.stop_reason}")
    text = next(b.text for b in r.content if b.type == "text")
    return {"result": json.loads(text),
            "usage": {"input": r.usage.input_tokens, "output": r.usage.output_tokens}}


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in MODELS or sys.argv[2] not in ("test", "send"):
        raise SystemExit(__doc__)
    name, which = sys.argv[1], sys.argv[2]
    model_id, p_in, p_out = MODELS[name]
    frames = frame_set(which)
    if len(sys.argv) > 3:
        frames = frames[:int(sys.argv[3])]
    cache = OUT / name
    cache.mkdir(parents=True, exist_ok=True)
    todo = [(i, p) for i, p in frames if not (cache / f"{i}.json").exists()]
    print(f"{model_id}: {len(frames)} frames, {len(todo)} to ask, {len(frames) - len(todo)} cached")

    client = anthropic.Anthropic(api_key=load_key(), max_retries=6)   # retries wait out rate limits
    failed = []
    with ThreadPoolExecutor(WORKERS) as pool:
        jobs = {pool.submit(ask, client, model_id, p): i for i, p in todo}
        for n, job in enumerate(as_completed(jobs), 1):
            fid = jobs[job]
            try:
                (cache / f"{fid}.json").write_text(json.dumps(job.result(), ensure_ascii=False, indent=1))
            except Exception as e:
                failed.append(fid)
                print(f"  {fid}: {e}")
            if n % 20 == 0:
                print(f"  {n} / {len(todo)}")

    # cost over every cached answer in this set
    used = [json.loads((cache / f"{i}.json").read_text())["usage"] for i, _ in frames if (cache / f"{i}.json").exists()]
    tin, tout = sum(u["input"] for u in used), sum(u["output"] for u in used)
    cost = tin / 1e6 * p_in + tout / 1e6 * p_out
    print(f"{len(used)} answers, {tin:,} input + {tout:,} output tokens = ${cost:.2f} "
          f"(${cost / max(len(used), 1):.4f} a frame). {len(failed)} failed")


if __name__ == "__main__":
    main()
