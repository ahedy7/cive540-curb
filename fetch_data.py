"""
CIVE 540 curb project: download Montreal open data and clip it to the study area.

Study area: Sherbrooke W (north), René-Lévesque O (south), Atwater (west), de Bleury (east).

Usage (from the project folder, with your virtual environment active):
    python fetch_data.py              # download everything, then clip
    python fetch_data.py --list       # show what each dataset contains, download nothing
    python fetch_data.py --only signs paid_parking
    python fetch_data.py --clip-only  # re-run clipping on files already downloaded

Output:
    data/raw/<dataset>/...            original files
    data/clipped/study_area.gpkg      one layer per spatial file, clipped (open in QGIS)
    data/clipped/tables/              non-spatial tables (e.g. paid-parking regulation codes)
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import requests

CKAN = "https://donnees.montreal.ca/api/3/action"
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
CLIPPED = ROOT / "data" / "clipped"
STUDY_AREA_FILE = ROOT / "study_area.geojson"
MTM8 = "EPSG:32188"  # NAD83 / MTM zone 8, the City's metric projection
BUFFER_M = 30        # keep features just outside the boundary streets

# key: (dataset slug or None, search text if slug is None, preferred formats, download all formats?)
DATASETS = {
    "signs":               ("stationnement-sur-rue-signalisation-courant", None, ["geojson", "csv"], False),
    "road_signs":          ("panneaux-de-signalisation", None, ["geojson", "csv"], False),
    "sign_poles":          (None, "poteaux de signalisation", ["geojson", "csv"], False),
    "paid_parking":        ("stationnements-municipaux-tarifes-sur-rue-et-hors-rue", None, ["csv"], False),
    "loading_zones":       (None, "aires de livraison sur rue", ["geojson", "shp", "csv"], False),
    "truck_rules":         ("camionnage-reglements", None, ["geojson", "shp"], False),
    "commercial_premises": ("locaux-commerciaux", None, ["geojson", "gpkg", "csv"], False),
    "collisions":          ("collisions-routieres", None, ["geojson", "csv"], False),
    "geobase":             ("geobase", None, ["geojson", "shp"], False),
    "geobase_double":      ("geobase-double", None, ["geojson", "shp"], False),
    "bike_network":        ("pistes-cyclables", None, ["geojson", "shp"], False),
    "pedestrian_streets":  (None, "rues piétonnes et partagées", ["geojson", "csv"], False),
    "property_units":      (None, "unités d'évaluation foncière", ["geojson", "csv"], False),
}

# Approximate corners of the study area (lon, lat). CHECK THESE: open study_area.geojson
# at https://geojson.io, drag the corners onto the four intersections, save over the file.
DEFAULT_STUDY_AREA = {
    "type": "FeatureCollection",
    "features": [{
        "type": "Feature",
        "properties": {"name": "CIVE 540 study area (approximate, verify)"},
        "geometry": {"type": "Polygon", "coordinates": [[
            [-73.5888, 45.4917],  # Atwater & Sherbrooke
            [-73.5718, 45.5105],  # de Bleury & Sherbrooke
            [-73.5640, 45.5050],  # de Bleury & René-Lévesque
            [-73.5830, 45.4868],  # Atwater & René-Lévesque
            [-73.5888, 45.4917],
        ]]},
    }],
}

session = requests.Session()
session.headers["User-Agent"] = "McGill-CIVE540-curb-project"


# ---------------------------------------------------------------- CKAN helpers
def ckan(action, **params):
    r = session.get(f"{CKAN}/{action}", params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    if not data.get("success"):
        raise RuntimeError(data.get("error"))
    return data["result"]


def get_package(slug, query):
    if slug:
        try:
            return ckan("package_show", id=slug)
        except Exception as e:
            print(f"   package_show failed for '{slug}' ({e}), falling back to search")
            query = slug.replace("-", " ")
    hits = ckan("package_search", q=query, rows=5)["results"]
    if not hits:
        raise RuntimeError(f"no dataset found for '{query}'")
    pkg = hits[0]
    print(f"   search '{query}' matched: {pkg['title']}  (slug: {pkg['name']})")
    if len(hits) > 1:
        others = ", ".join(h["name"] for h in hits[1:])
        print(f"   other matches, if that one is wrong: {others}")
    return pkg


def resource_format(res):
    fmt = (res.get("format") or "").lower().strip().lstrip(".")
    url = (res.get("url") or "").lower().split("?")[0]
    if fmt in ("", "zip"):
        for ext in ("geojson", "gpkg", "csv", "shp", "json"):
            if url.endswith("." + ext) or (ext == "shp" and url.endswith(".zip")):
                return ext
    if "shape" in fmt:
        return "shp"
    return fmt


def pick_resources(pkg, preferred):
    by_fmt = {}
    for res in pkg.get("resources", []):
        by_fmt.setdefault(resource_format(res), []).append(res)
    for fmt in preferred:
        if fmt in by_fmt:
            return fmt, by_fmt[fmt]
    return None, []


def download(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"   already have {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with session.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r   {dest.name}: {done / 1e6:.1f} / {total / 1e6:.1f} MB", end="")
        print()
    tmp.rename(dest)
    return dest


def safe_name(res, fmt):
    name = (res.get("name") or res["id"]).strip()
    name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:60]
    url_name = res["url"].split("?")[0].rsplit("/", 1)[-1]
    ext = Path(url_name).suffix or f".{ 'zip' if fmt == 'shp' else fmt }"
    return f"{name}{ext}"


def fetch(keys, list_only=False):
    manifest = {}
    for key in keys:
        slug, query, preferred, _ = DATASETS[key]
        print(f"\n[{key}]")
        try:
            pkg = get_package(slug, query)
        except Exception as e:
            print(f"   SKIPPED: {e}")
            continue
        if list_only:
            for res in pkg.get("resources", []):
                print(f"   - {resource_format(res):8} {res.get('name')}  {res.get('url')}")
            continue
        fmt, resources = pick_resources(pkg, preferred)
        if not resources:
            print(f"   SKIPPED: none of {preferred} available. Run with --list to see formats.")
            continue
        files = []
        for res in resources:
            try:
                files.append(str(download(res["url"], RAW / key / safe_name(res, fmt))))
            except Exception as e:
                print(f"   failed {res.get('name')}: {e}")
        manifest[key] = {"title": pkg["title"], "slug": pkg["name"], "format": fmt, "files": files}
    if not list_only:
        RAW.mkdir(parents=True, exist_ok=True)
        (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------- clipping
def study_area():
    import geopandas as gpd
    if not STUDY_AREA_FILE.exists():
        STUDY_AREA_FILE.write_text(json.dumps(DEFAULT_STUDY_AREA, indent=2))
        print(f"\nWrote approximate {STUDY_AREA_FILE.name}. Check the corners at geojson.io.")
    area = gpd.read_file(STUDY_AREA_FILE).to_crs(MTM8)
    return area.buffer(BUFFER_M).union_all()


def find_lonlat(columns):
    cols = {c.lower(): c for c in columns}
    for lon, lat in [("longitude", "latitude"), ("loc_long", "loc_lat"), ("lon", "lat"),
                     ("long", "lat"), ("nlongitude", "nlatitude"), ("x_long", "y_lat")]:
        if lon in cols and lat in cols:
            return cols[lon], cols[lat]
    lon = next((c for c in columns if "long" in c.lower()), None)
    lat = next((c for c in columns if "lat" in c.lower()), None)
    return (lon, lat) if lon and lat else (None, None)


def read_any(path):
    import geopandas as gpd
    import pandas as pd
    import zipfile
    suffix = path.suffix.lower()
    # check the content, not the extension: some City "geojson" downloads are really zips
    if zipfile.is_zipfile(path) and suffix != ".zip":
        # GDAL only opens zips named .zip, so pull the single inner file out in memory
        import io
        with zipfile.ZipFile(path) as z:
            inner = [n for n in z.namelist() if n.lower().endswith((".geojson", ".json"))]
            if len(inner) == 1:
                return gpd.read_file(io.BytesIO(z.read(inner[0])))
        raise ValueError(f"{path.name} is a zip without a single GeoJSON inside. Rename it to .zip")
    if zipfile.is_zipfile(path):
        try:
            return gpd.read_file(f"zip://{path}")
        except Exception:
            # the spatial file sits in a subfolder or next to other files: point at it directly
            with zipfile.ZipFile(path) as z:
                inner = [n for n in z.namelist() if n.lower().endswith((".shp", ".geojson", ".json"))]
            if not inner:
                raise
            return gpd.read_file(f"zip://{path}!{inner[0]}")
    if suffix in (".geojson", ".json", ".gpkg", ".shp"):
        return gpd.read_file(path)
    if suffix == ".csv":
        df = None
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                df = pd.read_csv(path, encoding=enc, low_memory=False)
                break
            except UnicodeDecodeError:
                continue
        lon, lat = find_lonlat(df.columns)
        if not lon:
            return df  # non-spatial table
        df[lon] = pd.to_numeric(df[lon], errors="coerce")
        df[lat] = pd.to_numeric(df[lat], errors="coerce")
        df = df.dropna(subset=[lon, lat])
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs="EPSG:4326")
    return None


def layer_name(key, stem, limit=60):
    """Short, ASCII, unique layer name. Keeps the END of the file name so years survive."""
    import re
    import unicodedata
    s = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    room = limit - len(key) - 2
    return f"{key}__{s[-room:]}" if len(s) > room else f"{key}__{s}"


def clip_all(keys):
    import geopandas as gpd
    area = study_area()
    CLIPPED.mkdir(parents=True, exist_ok=True)
    (CLIPPED / "tables").mkdir(exist_ok=True)
    gpkg = CLIPPED / "study_area.gpkg"
    if gpkg.exists():
        gpkg.unlink()  # rebuild from scratch so old layers don't linger
    summary = []
    for key in keys:
        folder = RAW / key
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix == ".part":
                continue
            print(f"\nclipping {key}/{path.name}")
            try:
                data = read_any(path)
            except Exception as e:
                print(f"   could not read: {e}")
                continue
            if data is None:
                continue
            if not isinstance(data, gpd.GeoDataFrame):
                shutil.copy(path, CLIPPED / "tables" / f"{key}__{path.name}")
                print(f"   non-spatial table, copied ({len(data)} rows)")
                summary.append((key, path.name, "table", len(data)))
                continue
            if data.crs is None:
                data = data.set_crs("EPSG:4326")
            data = data.to_crs(MTM8)
            data = data[data.intersects(area)].copy()
            layer = layer_name(key, path.stem)
            data.to_file(gpkg, layer=layer, driver="GPKG")
            print(f"   kept {len(data)} features -> layer {layer}")
            summary.append((key, path.name, layer, len(data)))
    print("\nSummary")
    for key, name, layer, n in summary:
        print(f"  {key:20} {n:>8}  {name}")
    print(f"\nOpen {gpkg.relative_to(ROOT)} in QGIS to check everything lines up.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list resources only")
    ap.add_argument("--only", nargs="+", choices=DATASETS.keys(), help="subset of datasets")
    ap.add_argument("--clip-only", action="store_true", help="skip downloading")
    args = ap.parse_args()
    keys = args.only or list(DATASETS)
    if not args.clip_only:
        fetch(keys, list_only=args.list)
    if not args.list:
        clip_all(keys)