# CIVE 540 Curb Mismatch Project: Downtown Montréal

This file is the full context for this project. Read it at the start of every session. Keep it updated: when something changes (a dataset is fixed, a decision is made, a phase is finished), edit the "Status" and "Decisions log" sections.

## Who I am and how to work with me

- I'm Arvin, a Statistics major at McGill (minors in Economics and Urban Studies). Strong in Python, statistics, GIS and QGIS. Not a professional software engineer, so keep code readable and explain non-obvious choices in one or two lines.
- Writing style for anything I'll read or submit: plain, direct language. No em dashes. No semicolons. Short sentences. No filler.
- Before writing a lot of code, say in one or two sentences what you're about to do. Then do it.
- Run things and check results yourself. Don't tell me something works without running it.
- When you hit a data problem, inspect the actual file (columns, CRS, a few rows) before guessing.
- Ask me before deleting data, overwriting raw files, or spending money on API calls over about $5.

## Environment

- MacBook Air (Apple Silicon), zsh, Homebrew, VS Code.
- There is no python3.12. Homebrew `python3` is actually 3.14 and does not have geopandas. The Python that has the geospatial stack is Anaconda: `/opt/anaconda3/bin/python3` (3.13, geopandas 1.1.3). If you set up a venv, use `python3 -m venv .venv`.
- ffmpeg is installed via Homebrew (verify with `ffmpeg -version`, install with `brew install ffmpeg` if missing).
- QGIS is installed. I check spatial outputs there.
- For YOLO or other models, use the Apple GPU through PyTorch MPS (`device="mps"`) when available.

## The project

- Course: CIVE 540 Urban Transportation Planning, McGill, Fall 2026. Instructor: Prof. Luis Miranda-Moreno. This is the final term project.
- Topic: curb management and urban delivery policies.
- Team of two. I cover the commercial and delivery side. My partner covers safety and livability.
- Study area: downtown Montréal bounded by Sherbrooke St W (north), René-Lévesque Blvd W (south), Atwater Ave (west), de Bleury St (east).
- Core idea: a curb mismatch analysis. Compare curb supply (how curb space is allocated, from City data checked by video) with delivery demand (modelled from businesses) and observed behaviour (occupancy, double parking, bike-lane blocking, from my own video). Produce a mismatch index per block face.
- Miranda-Moreno suggested using video plus vision-language models (VLMs) to read the curb, because curb data is scarce and costly.
- Final deliverable: a report, a presentation, and an interactive website where users adjust curb parameters and see the mismatch change.

## The goal beyond the class

This must be three things at once:
1. A strong final project.
2. A portfolio piece showing end-to-end spatial data science (data engineering, computer vision, statistics, web mapping). It will go on arvinhedayat.com and my resume. I'm targeting geospatial analytics and urban data science roles.
3. Something the City of Montréal and the Agence de mobilité durable could actually use.

So: clean public repo, reproducible pipeline, data in a standard curb format (CurbLR, mapped to the Open Mobility Foundation Curb Data Specification), honest accuracy numbers, and a one-page method note. Code quality and documentation matter.

Ethics and privacy review are not in scope for now (class project). Still, never commit raw unblurred video or frames to the public repo, and blur faces and plates in anything published on the website.

## Folder layout

```
cive540-curb/
  CLAUDE.md
  fetch_data.py          # downloads City open data and clips to study area
  requirements.txt
  study_area.geojson     # study area, traced from géobase centre lines of the 4 boundary streets
  data/
    raw/<dataset>/       # original downloads, never edit
    clipped/study_area.gpkg   # one layer per dataset, EPSG:32188
    clipped/tables/      # non-spatial tables (paid parking regulations etc.)
  video/                 # (to create) raw iPhone clips + GPX tracks, gitignored
  frames/                # (to create) extracted frames, gitignored
  pipeline/              # (to create) video processing scripts
  analysis/              # (to create) supply, demand, metrics, models
  web/ or docs/          # (to create) MapLibre site for GitHub Pages
```

Add a `.gitignore` covering `data/raw/`, `video/`, `frames/`, `.venv/`, large files, and `.DS_Store`.

## Coordinate systems

- Work and measure in EPSG:32188 (NAD83 / MTM zone 8, metres). This is the City's projection.
- Export to EPSG:4326 only for the website.

## Data: what's downloaded

`fetch_data.py` pulls these from donnees.montreal.ca (CKAN API) and clips to the study area with a 30 m buffer. Counts are from the Nov 2 re-clip with the corrected study area. Layer names in `study_area.gpkg` are ASCII and shown below.

| Key | GPKG layer(s) | Clipped count | Notes |
|---|---|---|---|
| signs | signs__Signalisation_Stationnement | 3,315 | Every parking sign with RPA code. Main supply layer. Join to poles via POTEAU_ID_POT |
| road_signs | road_signs__Signalisation_routiere | 4,929 | Stops, one-ways, bus stops etc. |
| sign_poles | sign_poles__Poteaux_de_signalisation | 2,666 | |
| paid_parking | paid_parking__Places, paid_parking__BornesSurRue, paid_parking__BornesHorsRue | 2,354 / 144 / 0 | Regulation tables (Réglementations, EmplacementReglementation, ReglementationPeriode, Periodes) are in data/clipped/tables. Must credit "Agence de mobilité durable" (CC BY 4.0). BornesHorsRue is empty (off-street) |
| loading_zones | loading_zones__Aires_de_livraison | 62 | Points. Zip with shapefile in a subfolder |
| truck_rules | truck_rules__Format_GeoJSON, truck_rules__Zones_d_interdiction | 414 + 9 | |
| commercial_premises | commercial_premises__quete_sur_l_occupation_commerciale_2021 … _2025 | 1,489 / 1,568 / 1,627 / 1,629 / 1,632 | Main demand driver. Use 2025. Includes VACANT units |
| collisions | collisions__Collisions_routieres, collisions__Polygones_d_analyses_aux_intersections | 6,947 + 169 | Partner's safety layer. The polygons file is a zip saved as .geojson |
| geobase | geobase__Geobase | 424 | Street centrelines. Build block faces from this |
| geobase_double | geobase_double__Geobase_double | 832 | One line per street side (COTE, COT_RUE_ID). May be easier for block faces |
| bike_network | bike_network__Reseau_cyclable | 38 | |
| pedestrian_streets | pedestrian_streets__Liste_des_rues_pietonnes | 2 | Points only |
| property_units | property_units__eographique_des_unites_d_evaluation_fonciere | 14,551 | Floor area and use codes. Demand proxy |

Also downloaded but not yet used: the RPA and RTP sign code tables (CSV) inside the signs dataset, sign images, the 2015 sign catalogue PDF.

Requested by email (may or may not arrive): the Agence's Ville-Marie curb inventory (camera van survey), Agence occupancy and transaction data, Miranda-Moreno's class data.

Other sources to consider later: STM GTFS (bus stops), BIXI stations, OpenStreetMap (shops, buildings), Mapillary API (gap filling, CC BY-SA).

## Status (update this)

- [x] fetch_data.py written and run once. Most data downloaded and clipped.
- [x] Re-run `python fetch_data.py --clip-only` with the fixed script. All datasets now clip, including signs, loading zones, all commercial years and collision polygons (Nov 2)
- [x] Fix study_area.geojson. Now traced from the géobase centre lines of the 4 boundary streets (1.405 km2). Re-clipped (Nov 2)
- [x] Verify clipped layers in QGIS. Boundary and layers line up with OSM (Nov 3)
- [ ] Build block faces with stable IDs
- [ ] Filming (see plan)
- [ ] Video pipeline
- [ ] Analysis
- [ ] Website
- [ ] Report and presentation

## Timeline

Deadline assumed around Dec 7 to 11, 2026 (confirm). Filming is front-loaded before the weather gets worse. Sunset is around 4:15 pm.

- **Nov 2 to 8:** setup, data, route planning. Pilot ride Wednesday (10 to 15 block faces on de la Montagne and Ste-Catherine near Peel). Full inventory pass Thursday to Saturday, 10:00 to 14:00, every block face once.
- **Nov 9 to 15:** morning activity passes Tue, Wed, Thu 08:00 to 11:00 on the delivery core (Ste-Catherine, de Maisonneuve, cross streets from Guy to Robert-Bourassa). Fixed-point filming at 3 to 5 loading zones, 60 to 90 min each, for dwell time. Sunday coverage check.
- **Nov 16 to 22:** reshoots, build and run the pipeline, join sign inventory to block faces, hand-label 300 frames.
- **Nov 23 to 29:** analysis (supply, demand, observed metrics, mismatch index, model).
- **Nov 30 to Dec 6:** website and report draft.
- **Dec 7 to 11:** finish and submit.

## Video capture (iPhone, no GoPro)

- iPhone, 4K at 30 fps, High Efficiency format, 1x lens (not 0.5x ultrawide), focus and exposure locked.
- Handlebar mount angled 30 to 45 degrees toward the curb. Walk with the phone on one-way streets and dense sign clusters.
- iPhone video stores only one location per clip, not a track. A separate GPX logger app (Open GPX Tracker or Strava) records the track. Sync frames to GPS by timestamp.
- Clip timestamps: read `creation_time` (and Apple's `com.apple.quicktime.creationdate`) with ffprobe. Watch for time zone offsets. GPX is in UTC. Allow a manual offset per clip in case the clocks don't line up.
- Clips are .MOV (HEVC). ffmpeg handles them.

## Video pipeline spec (to build in pipeline/)

1. **Ingest:** list clips in video/, read start time and duration with ffprobe, pair each clip with its GPX file. Save a clips manifest CSV.
2. **Frames:** extract 2 frames per second with ffmpeg (`-vf fps=2 -q:v 2`). Record each frame's timestamp = clip start + index / 2.
3. **Geotag:** interpolate lat/lon and heading from the GPX track for each frame timestamp.
4. **Snap to block faces:** nearest géobase segment within 15 m (in EPSG:32188). Use heading and which side the camera faces (right by default, configurable per clip) to pick the block face side. Drop frames within about 10 m of intersections.
5. **Blur:** faces and licence plates with EgoBlur (`pip install egoblur`) on frames that will be published. Analysis can run on unblurred frames locally.
6. **Filter:** run Ultralytics YOLO11 (pretrained COCO) on all frames. Keep frames with a curbside vehicle, a sign-like object, or a hydrant. Cuts VLM calls a lot.
7. **VLM extraction:** send filtered frames (or sign crops) to a vision-language model with a fixed JSON schema. Use the Anthropic API (Claude) unless I say otherwise. Run a 50 to 100 frame sample first and report cost before the full run. Use batching if available. Cache every response to disk keyed by frame ID so nothing is paid for twice. Schema:
   ```json
   {"signs":[{"type":"no_parking|no_stopping|paid|loading|reserved|other",
     "text_fr":"", "hours":"", "days":"", "arrow":"left|right|both|none"}],
    "curb_features":["hydrant","driveway","bus_stop","bike_lane","yellow_paint"],
    "vehicles":[{"class":"car|delivery_van|box_truck|pickup|cargo_bike|bus|taxi",
     "position":"parked_curb|double_parked|in_bike_lane|in_bus_stop|moving",
     "loading_activity":true}],
    "confidence":0.0}
   ```
   Include a few RPA code descriptions in the prompt so the model maps French sign text ("Livraison seulement", "Arrêt interdit", "Stationnement interdit") to the categories.
8. **Validation:** a labelling helper (simple local HTML page or notebook) for hand-labelling 300 stratified frames. Compute precision, recall and F1 per class and Cohen's kappa on 50 double-labelled frames.
9. **Reconcile:** compare video sign reads with City sign records within 10 m. Label each as confirmed, missing in data, missing on street, or conflicting.

## Analysis spec (to build in analysis/)

- **Unit:** block face (one side of a street between two intersections) by time window (07 to 11, 11 to 15, 15 to 19, weekdays).
- **Block faces:** from géobase or géobase double. Stable ID like `<street>_<from>_<to>_<L|R>`. Store length in metres.
- **Supply:** linear metres per block face by use and time window: paid parking, unpaid or permit, loading, no stopping or no parking, implicit no parking (hydrants, driveways, intersection clearances), bus stop, reserved or taxi, other. Parse RPA descriptions into use, hours and days. Paid spaces from Places.csv at about 6 m each if no length is given.
- **Demand:** 2025 commercial premises, occupied only. Map use categories to groups (food retail, restaurants, general retail, services, offices). Daily freight trips per establishment from NCFRP Report 37 constant-per-establishment rates. Spread over the day with a peak share in 08 to 11. Assign to the frontage block face (or nearest legal block face within 50 to 75 m if on a pedestrian street).
- **Required loading length:** peak arrivals per hour, mean dwell (from fixed-camera data or 10 to 20 min default), space length 9 to 12 m. Erlang-B / M/M/c so P(all busy) < 10 percent. Required length = spaces × length.
- **Observed:** occupancy share across passes, misuse rate of loading zones, double-parking rate per 100 m per pass, bike-lane blockage rate, dwell and turnover at fixed sites.
- **Mismatch index:** MI = (required − supplied) / max(required, supplied), from −1 (oversupply) to +1 (undersupply).
- **Model:** negative binomial GLM of double-parking counts on MI, commercial density, bike-lane presence, street type, with block-face random effects or clustered errors. Report rate ratios. This is the key validation that the demand model means something.
- **Partner's side (support when asked):** collisions within 25 to 30 m of loading activity, bike-lane blocking, sidewalk width samples, curb share for people versus vehicle storage.

## Website spec (to build in docs/ for GitHub Pages)

- MapLibre GL JS, plain HTML/JS, no build step. Single `index.html`, `app.js`, `style.css`, `blockfaces.geojson`.
- Map coloured by mismatch index. Layer toggles: regulations, demand, observed double parking, collisions, bike lanes.
- Sliders: convert N paid spaces per block face to loading, loading time window, dwell time, vehicle length, share of deliveries consolidated to cargo bikes, off-peak delivery share. Recompute MI in the browser.
- Summary panel: block faces in undersupply, loading metres added, parking spaces and estimated revenue lost (rates from Places.csv), estimated change in double parking.
- 3 or 4 preset scenarios.
- Distinctive, clean design. It must not look like a generic template or AI-generated. Works on phone width.
- Data per block face follows a CurbLR-aligned schema, with a mapping table to CDS field names.

## Repo and portfolio requirements

- README with: what the project is, a screenshot, the live link, how to reproduce (commands in order), data sources and licences with attribution, accuracy results, limitations.
- A pipeline diagram.
- A blurred sample of frames and outputs.
- Clean commit history with clear messages.

## Decisions log (add to this)

- Nov 2: iPhone instead of GoPro. GPS from a separate GPX app.
- Nov 2: ethics and privacy review out of scope for now. Still no raw video in the repo and blur anything published.
- Nov 2: video is used to verify the City sign inventory and measure behaviour, not to rebuild regulations from scratch.
- Nov 2: commercial premises 2025 is the main demand year.
- Nov 2: fetch_data.py checks file content, not extension, for zips (some City .geojson downloads are zips). GPKG layer names are ASCII and keep the end of the file name so years stay distinct.
- Nov 2: study area boundary follows the actual street centre lines, not 4 straight corners. René-Lévesque curves up to 130 m and Sherbrooke up to 84 m from a straight line, so a corner polygon cut off or added whole blocks.
- Nov 3: filming moved to the week of Nov 9. Computer work (block faces, supply, demand) comes first. Timeline below needs shifting.
