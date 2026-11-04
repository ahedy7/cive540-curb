"""
Turn each parking sign rule (RPA code) into a use, a schedule and a season.

Input:  data/clipped/study_area.gpkg, layer signs__Signalisation_Stationnement (active signs only)
Output: analysis/rpa_rules_draft.csv, one row per rule, for review by hand.

The reviewed copy is saved as analysis/rpa_rules.csv. That file is the one supply.py uses.
This script never touches it, so hand edits are safe.

Usage:
    python analysis/rpa_rules.py
"""
import re
import unicodedata
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GPKG = ROOT / "data" / "clipped" / "study_area.gpkg"
OUT = ROOT / "analysis" / "rpa_rules_draft.csv"

DAY_WORDS = {"LUNDI": "Mo", "MARDI": "Tu", "MERCREDI": "We", "JEUDI": "Th", "VENDREDI": "Fr",
             "SAMEDI": "Sa", "DIMANCHE": "Su", "LUN": "Mo", "MAR": "Tu", "MER": "We", "JEU": "Th",
             "VEN": "Fr", "SAM": "Sa", "DIM": "Su"}
WEEK = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
MONTHS = {"JANV": 1, "FEV": 2, "MARS": 3, "AVRIL": 4, "MAI": 5, "JUIN": 6, "JUIL": 7,
          "AOUT": 8, "SEPT": 9, "OCT": 10, "NOV": 11, "DEC": 12}

# who a reserved or excepted space is for. First match wins, so specific phrases come first
BENEFICIARY = [
    ("LIVRAISON", "delivery"), ("S3R", "resident_permit"), ("TAXI", "taxi"), ("HOTEL", "hotel"),
    ("HANDICAPES DEBARCADERE", "disabled_dropoff"), ("HANDICAPE", "disabled"),
    ("VOITURE ELECTRIQUE", "ev_charging"), ("MOTO", "motorcycle"),
    ("VISITEURS POSTE DE POLICE", "police_visitors"), ("POLICE", "police"),
    ("CONSULAIRES", "diplomatic"), ("AUTOBUS TOURISTIQUE", "tour_bus"), ("AUTOBUS", "bus"),
    ("MILITAIRE", "military"), ("INCENDIE", "fire"), ("POMPIER", "fire"),
    ("VEHICULES DE LA VILLE", "city"), ("GARDERIE", "daycare"),
]


def plain(text):
    """Upper case, no accents, single spaces."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().upper()
    return re.sub(r"\s+", " ", text).strip()


def hhmm(h, m):
    return f"{int(h):02d}:{int(m or 0):02d}"


TIME_RE = re.compile(r"(\d{1,2})(?:H(\d{2})?)?\s*-\s*(\d{1,2})H?(\d{2})?(?!\s*(?:AVRIL|DEC|OCT|NOV|MARS))")
SEASON_RE = re.compile(r"(\d{1,2}) (" + "|".join(MONTHS) + r")\w* AU (\d{1,2}) (" + "|".join(MONTHS) + r")")


def parse_days(text):
    """Days named in a clause, e.g. 'LUN A VEN' -> Mo-Fr, 'LUNDI JEUDI' -> Mo,Th."""
    text = text.replace(".", " ")
    rng = re.search(r"\b(LUN|MAR|MER|JEU|VEN|SAM|DIM)\w*\s+(?:A|AU)\s+(LUN|MAR|MER|JEU|VEN|SAM|DIM)\w*", text)
    if rng:
        a, b = WEEK.index(DAY_WORDS[rng.group(1)]), WEEK.index(DAY_WORDS[rng.group(2)])
        return WEEK[a:b + 1]
    found = [DAY_WORDS[w] for w in re.findall(r"\b(" + "|".join(sorted(DAY_WORDS, key=len, reverse=True)) + r")\b", text)]
    return sorted(set(found), key=WEEK.index)


def day_label(days):
    if not days or days == WEEK:
        return "Mo-Su"
    if days == WEEK[days.index(days[0]):days.index(days[0]) + len(days)] and len(days) > 2:
        return f"{days[0]}-{days[-1]}"
    return ",".join(days)


def parse_schedule(text):
    """
    'LUN A VEN 18h-23h ... 9h-23h SAM DIM' -> 'Mo-Fr 18:00-23:00 | Sa,Su 09:00-23:00'.
    Times with no days named apply every day. Returns (schedule, needs_review).
    """
    if "EN TOUT TEMPS" in text or not TIME_RE.search(text):
        return "always", False
    season = SEASON_RE.search(text)
    body = text[:season.start()] + text[season.end():] if season else text
    times = list(TIME_RE.finditer(body))
    # split into clauses: each clause = some times followed by its days (or days then times)
    day_spans = [m for m in re.finditer(r"\b(LUN|MAR|MER|JEU|VEN|SAM|DIM)\w*\.?(?:\s+(?:A|AU|ET)?\s*(?:LUN|MAR|MER|JEU|VEN|SAM|DIM)\w*\.?)*", body)]
    if len(day_spans) <= 1:
        days = parse_days(body)
        slots = [f"{hhmm(t.group(1), t.group(2))}-{hhmm(t.group(3), t.group(4))}" for t in times]
        return f"{day_label(days)} " + ",".join(slots), False
    # several day groups (e.g. weekdays one way, weekends another). On these signs the days
    # come after their times ("18h-23h LUN A VEN 9h-23h SAM DIM"), so pair each time with the
    # next day group, or the last one if none follows
    parts = []
    for t in times:
        after = [d for d in day_spans if d.start() >= t.end()]
        grp = after[0] if after else day_spans[-1]
        parts.append((day_label(parse_days(grp.group(0))), f"{hhmm(t.group(1), t.group(2))}-{hhmm(t.group(3), t.group(4))}"))
    merged = {}
    for d, slot in parts:
        merged.setdefault(d, []).append(slot)
    return " | ".join(f"{d} {','.join(s)}" for d, s in merged.items()), True


def parse_season(text):
    m = SEASON_RE.search(text)
    if not m:
        return "all_year"
    return f"{MONTHS[m.group(2)]:02d}-{int(m.group(1)):02d}..{MONTHS[m.group(4)]:02d}-{int(m.group(3)):02d}"


def classify(code, raw):
    """Returns (use, beneficiary, time_limit_min, needs_review, note)."""
    t = plain(raw)
    ben = next((b for key, b in BENEFICIARY if key in t), "")
    minutes = re.search(r"\b(\d+) ?MIN", t)
    hours = re.match(r"P 0?(\d+) ?H\b", t)  # "P 2h 9h-23h": only at the start, so 9h-23h isn't read as a limit
    limit_min = int(minutes.group(1)) if minutes else (int(hours.group(1)) * 60 if hours else None)

    if code == "PP-AQ" or "TARIF" in t:
        return "paid_zone", "", None, False, "Cross-check only. Paid metres come from Places.csv"
    if t.startswith("\\A"):
        if "RESERVE AUTOBUS" in t:
            return "bus_stop", "bus", None, False, ""
        if ben == "fire":
            return "no_stopping", "fire", None, False, "Fire lane"
        if ben:
            return "reserved", ben, None, False, "No stopping except this group"
        return "no_stopping", "", None, False, ""
    if t.startswith("\\P"):
        if "EXCEPTE" in t or "RESERVE" in t:
            if ben == "delivery":
                return "loading", "delivery", None, False, ""
            if ben == "resident_permit":
                return "permit", "resident_permit", None, False, ""
            if ben:
                return "reserved", ben, limit_min, False, ""
            return "reserved", "", limit_min, True, "Exception group not recognised"
        note = "Street cleaning" if SEASON_RE.search(t) else ""
        if "DEUX COTES" in t:
            note = "Sign says both sides of the street"
        return "no_parking", "", None, False, note
    if t.startswith("P "):
        if ben == "bus":
            return "reserved", "bus", limit_min, False, ""
        if ben:
            return "reserved", ben, limit_min, False, "Short-term, reserved for this group"
        return "short_term", "", limit_min, limit_min is None, ""
    return "unknown", ben, limit_min, True, "Pattern not recognised"


def main():
    s = gpd.read_file(GPKG, layer="signs__Signalisation_Stationnement", columns=["CODE_RPA", "DESCRIPTION_RPA", "DESCRIPTION_REP"])
    s = s[s.DESCRIPTION_REP == "Réel"]
    rules = s.groupby(["CODE_RPA", "DESCRIPTION_RPA"]).size().reset_index(name="n_signs")

    rows = []
    for r in rules.itertuples():
        use, ben, limit, review, note = classify(r.CODE_RPA, r.DESCRIPTION_RPA)
        sched, sched_review = parse_schedule(plain(r.DESCRIPTION_RPA))
        rows.append({
            "code_rpa": r.CODE_RPA, "sign_text": r.DESCRIPTION_RPA, "n_signs": r.n_signs,
            "use": use, "beneficiary": ben, "schedule": sched,
            "season": parse_season(plain(r.DESCRIPTION_RPA)), "time_limit_min": limit,
            "needs_review": "yes" if (review or sched_review) else "",
            "note": note,
        })
    out = pd.DataFrame(rows).sort_values(["needs_review", "n_signs"], ascending=[False, False])
    out["time_limit_min"] = out.time_limit_min.astype("Int64")
    out.to_csv(OUT, index=False, encoding="utf-8-sig")  # utf-8-sig so Excel shows accents
    print(f"{len(out)} rules -> {OUT.relative_to(ROOT)}")
    print(out.use.value_counts().to_string())
    print(f"flagged for review: {(out.needs_review == 'yes').sum()}")


if __name__ == "__main__":
    main()
