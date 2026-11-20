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
- Team of two. I cover the commercial and delivery side. My partner covers safety and livability, plus validation labelling. I build the website.
- Study area: downtown Montréal bounded by Sherbrooke St W (north), René-Lévesque Blvd W (south), Atwater Ave (west), de Bleury St (east).
- **Headline question:** where does downtown lack loading space, does that predict double parking, and how much curb would fix it. Everything serves this. The sign check (City records vs street) is a side finding.
- Core idea: a curb mismatch analysis. Compare curb supply (how curb space is allocated, from City data checked by video) with delivery demand (modelled from businesses) and observed behaviour (occupancy, double parking, bike-lane blocking, from my own video). Produce a mismatch index per block face.
- Miranda-Moreno suggested using video plus vision-language models (VLMs) to read the curb, because curb data is scarce and costly.
- Final deliverable: a report, a presentation, and an interactive website where users adjust curb parameters and see the mismatch change.

## The goal beyond the class

This must be three things at once:
1. A strong final project.
2. A portfolio piece showing end-to-end spatial data science (data engineering, computer vision, statistics, web mapping). It will go on arvinhedayat.com and my resume. I'm targeting geospatial analytics and urban data science roles.
3. Something the City of Montréal and the Agence de mobilité durable could actually use.

So: clean public repo, reproducible pipeline, data in our own simple, clearly named schema, honest accuracy numbers, and a one-page method note. An Open Mobility Foundation CDS export is an optional extra at the very end, only if time allows. Code quality and documentation matter.

Ethics and privacy review are not in scope for now (class project). Still, never commit raw unblurred video or frames to the public repo, and blur faces and plates in anything published on the website.

## Folder layout

```
cive540-curb/            # lives at ~/cive540-curb (moved off the iCloud Desktop on Nov 16)
  CLAUDE.md
  fetch_data.py          # downloads City open data and clips to study area
  requirements.txt
  study_area.geojson     # study area, traced from géobase centre lines of the 4 boundary streets
  data/
    raw/<dataset>/       # original downloads, never edit
    clipped/study_area.gpkg   # one layer per dataset, EPSG:32188
    clipped/tables/      # non-spatial tables (paid parking regulations etc.)
    derived/blockfaces.gpkg   # block faces (analysis/blockfaces.py)
    derived/curb_supply.gpkg  # curb stretches + supply per block face and window (analysis/supply.py)
    derived/premises_2025.gpkg  # 2025 premises matched to block faces (analysis/premises.py)
    derived/sample.gpkg       # stratified sample (analysis/sample.py), also sample.csv
    derived/demand.gpkg       # deliveries per premises and block face, low/high (analysis/demand.py)
    derived/loading.gpkg      # Erlang loading model + mismatch per face and window (analysis/loading.py), also mismatch.csv
    derived/maps/             # mismatch_07-11.png
    derived/route/            # car loop: route.gpx, route.gpkg, turn_by_turn.txt/.csv, car_loop.png (analysis/route.py)
  video/                 # raw iPhone clips (original names, e.g. IMG_4730.MOV) + GPX tracks, gitignored
  frames/<file stem>/    # 1 fps frames, 000001.jpg ..., gitignored
  pipeline/              # common.py, ingest.py (-> clips.csv manifest), frames.py, geotag.py (-> data/derived/frames.gpkg, coverage.csv)
  analysis/              # supply, demand, metrics, models. blockfaces.py done
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
| hydrants | hydrants__Bornes_d_incendie | 220 | All active. City of Montréal, CC BY 4.0. Used for 5 m no-parking around each hydrant |

Also downloaded but not yet used: the RPA and RTP sign code tables (CSV) inside the signs dataset, sign images, the 2015 sign catalogue PDF.

Requested by email (may or may not arrive): the Agence's Ville-Marie curb inventory (camera van survey), Agence occupancy and transaction data, Miranda-Moreno's class data.

Other sources to consider later: STM GTFS (bus stops), BIXI stations, OpenStreetMap (shops, buildings), Mapillary API (gap filling, CC BY-SA).

## Status (update this)

- [x] fetch_data.py written and run once. Most data downloaded and clipped.
- [x] Re-run `python fetch_data.py --clip-only` with the fixed script. All datasets now clip, including signs, loading zones, all commercial years and collision polygons (Nov 2)
- [x] Fix study_area.geojson. Now traced from the géobase centre lines of the 4 boundary streets (1.405 km2). Re-clipped (Nov 2)
- [x] Verify clipped layers in QGIS. Boundary and layers line up with OSM (Nov 3)
- [x] Build block faces with stable IDs. 522 faces, 49.0 km of curb, data/derived/blockfaces.gpkg (Nov 4)
- [x] Task 1, curb supply (Nov 4). analysis/supply.py -> data/derived/curb_supply.gpkg (stretches, supply_by_window) + supply_by_window.csv
  - [x] Hydrant data found and added (220). Driveways: no usable City data (sidewalk value list is access-denied), so video-only on sampled faces, a limitation
  - [x] Paid parking source chosen: Places.csv + regulation tables. The 11 "STATIONNEMENT TARIFÉ" signs are a cross-check only
  - [x] Arrow codes: 2 = left, 3 = right, 0 = no arrow, 8 = both. Inferred from 929 same-rule sign pairs (about 80% agreement) and confirmed by the Agence's signalec_to_curblr code, prkng and Savoir-faire Linux's converter. Street View was not usable (construction, trucks)
  - [x] Rules reviewed and approved by Arvin as drafted: analysis/rpa_rules.csv (130 rules)
  - [x] Stretch file (4,785 stretches) + summary table (522 faces x 3 windows). Uses sum to face length within 0.2 m. 50 of 62 official loading zones sit on a loading stretch
- [x] Task 2, sample and route (Nov 4). 62 faces in 32 blocks. analysis/premises.py, sample.py -> data/derived/sample.gpkg + sample.csv
- [x] Car route re-planned (Nov 7). One loop, all 62 faces, 15.8 km, 47 to 63 min at 20 to 15 km/h. analysis/route.py -> data/derived/route/ (route.gpx, turn_by_turn.txt/.csv, car_loop.png, route.gpkg with turn_bans layer)
- [x] Delivery demand (Nov 7). analysis/demand.py + delivery_rates.csv -> data/derived/demand.gpkg. 983 curb premises, 970 (low) to 1,919 (high) deliveries a weekday
- [x] Loading model and first mismatch map (Nov 10). analysis/loading.py -> data/derived/loading.gpkg, mismatch.csv, maps/mismatch_07-11.png. 07-11 peak: 126 to 150 faces clearly short, 1.4 to 2.2 km of curb short. 23 to 29% of the day's trucks find no space
  - [x] Placeholders (Nov 18): analysis/occupancy.py (Haiku + before/after on 631 side-camera frames, $0.41). VACANT_SHARE now 0.30 (video reads 0.51, an upper bound, range 0.15 to 0.50). MISUSE_SHARE stays 0.30 as an assumption. 07-11 now: 92 to 118 faces short, 1.1 to 1.7 km of curb
  - [x] Arvin confirmed the flagged rates in delivery_rates.csv (hotels, offices) and leaving mall units off the curb (Nov 18)
- [ ] Weekend test drive (Nov 7 to 8): both mounts, GPX sync, sign readability, check the route's turn bans
- [x] Filming done: Mon Nov 9, Wed Nov 11, Fri Nov 13. No backup drive (no time)
- [ ] Video pipeline
  - [x] Steps 1 to 4 written (Nov 13): ingest, frames, geotag, snap. Any file names, split clips, any GPX app, manual offset per clip
  - [x] All frames extracted (Nov 16). 21 clips in pipeline/clips.csv: IMG_4723-4734 (Arvin, forward camera F, days 1 and 2, 1080p) and the partner's Monday/Wednesday/Friday left (L, iPhone 16 Pro) and Friday right (R, iPhone 15) clips (720p exports). All videos deleted after frame counts matched. Partner originals on OneDrive
  - [x] Partner exports lost the recording time. Real starts read off the phone's Photos info panel, in pipeline/start_times.csv. ingest.py reads it and keeps clips whose videos are deleted
  - [x] pipeline/sync.py: per-clip offset by matching frame-to-frame motion with GPS speed. Tested on 20261113_R_1: +1 s, Spearman 0.79. Run on all clips once their frames are local
  - [x] Coverage (Nov 16, Wed + Fri): 43 of 62 sample faces seen on 2 days, 13 on 1 day, 6 on none (construction, see decisions)
  - [x] Monday and early Friday geotagged from the partner's route log (Nov 17). pipeline/route_log.csv -> pipeline/anchors.py -> video/<date>_route_log.csv. Forward camera synced to it within 1 s. Coverage now: Monday 46, Wednesday 53, Friday 54 of 62 faces. All days: 60 of 62 (only Sainte-Catherine Peel-Metcalfe N/S never filmed). Timing check on L clips with GPS: median error 13 m, 90th pct 47 m (constant speed: 28 m and 103 m)
  - [ ] YOLO, VLM and review list on the new Monday frames
  - [x] Labelling page + 300 validation frames sent to the partner (validation/curb_labelling.zip). Only the first 100 are needed (Nov 18). Labels pending
  - [x] YOLO pre-filter (Nov 16): pipeline/detect.py. 2,397 frames on sample faces, 1,248 sent to the VLM (near vehicle or hydrant, one frame per 10 s while stopped)
  - [x] VLM (Nov 16): pipeline/vlm.py. 100-frame test on Haiku and Sonnet 5.5, with and without before/after frames. Full run: Haiku 5.5 + before/after on all 1,248 frames ($1.18). About $3 of API spent in total
  - [x] Partner reviewed double parking part 1 (Nov 18): all 318 frames, validation/review_partner_2026-11-18.json. 37 double parked frames (about 10 events)
  - [x] Partner: part 2 (Nov 20), all 531 frames in validation/review_partner_2026-11-20.json. Monday all no, one new event (Fri, Sainte-Catherine Metcalfe-Mansfield, 3 vehicles)
  - [x] Double-parking events and first model (Nov 20). analysis/double_parking.py -> dp_events.csv, dp_blocks.csv, dp_model.csv. 9 events on 7 of 31 filmed blocks, 14.2 km of curb passed, 0.63 events per km. MI and P(no space) do not predict double parking (RR 1.04 per +0.1 MI, CI 0.96 to 1.12). Deliveries a day do (RR 1.5 per doubling, CI 1.1 to 2.1, p 0.01). Blocks with 9+ deliveries a day: 1.39 per km vs 0.22. Trucks with no space (arrivals x P(no space), per block, log2): RR 1.3 per doubling (CI 0.9 to 1.9), and 1.03 (CI 0.7 to 1.5) once deliveries are in the model
  - [ ] Score the VLM against the partner's labels (precision, recall, F1, kappa)
- [ ] Analysis
- [ ] Website
  - [x] Look chosen (Nov 20): planning document (light, editorial), built at docs/index.html. docs/model.js runs the loading model in the browser and matches loading.py on all 522 faces. Data from analysis/web_data.py. Screenshot for slide 12: figures/website.png
  - [ ] Turn on GitHub Pages (Settings > Pages > main /docs). Repo is private, so Pages needs GitHub Pro (free with the Student Pack) or a public repo. The site itself is public once on
  - [ ] Revenue lost (Places.csv rates), per-block off-peak shifts, click a face for details
- [ ] Report and presentation

## Timeline

Presentation of the whole project on Wed Nov 25, 2026. Final website and report due later (date to confirm, assumed Dec 7 to 11). Sunset is around 4:15 pm.

- **Nov 4 to 8:** computer work: curb supply (Task 1), stratified sample (Task 2), car route, demand.
- **Weekend Nov 7 to 8:** test drive of the full loop. Replaces the pilot ride. Check both mounts, GPX sync, sign readability, and the route's turns.
- **Mon Nov 9, Wed Nov 11, Fri Nov 13:** car passes, leaving Peel and Ste-Catherine at 08:15. Backup Mon Nov 16 if one fails.
- **Nov 13 to 24:** build and run the video pipeline, validation set, sign check.
- **Nov 25 to Dec 1:** analysis (supply, demand, observed metrics, mismatch index, model).
- **Dec 2 to 6:** website and report draft.
- **Dec 7 to 11:** finish and submit.

## Video capture (iPhone, no GoPro)

- Scope: 3 weekday morning car passes on the sample route, 62 block faces. No full inventory. No fixed-camera sessions. No hand-held footage.
- By car. The partner drives, Arvin navigates from the passenger seat with data/derived/route/turn_by_turn.txt and car_loop.png (or route.gpx).
- Two mounted iPhones:
  - **R camera:** front passenger window, films the right curb. Used on all streets.
  - **L camera:** rear driver-side window, films the left curb. Used only on one-way streets. On two-way streets its frames show the far curb across traffic and are dropped.
- One loop per morning from Peel and Ste-Catherine, 15.8 km, 47 to 63 min at 20 to 15 km/h average. One-way blocks are filmed on both sides in one pass. Two-way blocks get one pass in each direction with the face on the right. Drive slowly and steadily on the steps marked SAMPLE.
- iPhone, 4K at 30 fps, High Efficiency format, 1x lens (not 0.5x ultrawide), focus and exposure locked.
- Each phone runs its own GPX logger app (Open GPX Tracker or Strava). iPhone video stores only one location per clip, not a track. Sync frames to GPS by timestamp.
- One clip per phone per morning, both recording the whole loop. File names: `YYYYMMDD_R.MOV` + `YYYYMMDD_R.gpx` and `YYYYMMDD_L.MOV` + `YYYYMMDD_L.gpx`.
- Clip timestamps: read `creation_time` (and Apple's `com.apple.quicktime.creationdate`) with ffprobe. Watch for time zone offsets. GPX is in UTC. Allow a manual offset per clip in case the clocks don't line up.
- Clips are .MOV (HEVC). ffmpeg handles them.

## Video pipeline spec (to build in pipeline/)

1. **Ingest:** list clips in video/, read start time and duration with ffprobe, pair each clip with its GPX file by name (`YYYYMMDD_R.MOV` with `YYYYMMDD_R.gpx`). The date and the camera side (R or L) come from the file name. Save a clips manifest CSV with date, side, start, duration, GPX path and a manual time offset column.
2. **Frames:** extract 1 frame per second with ffmpeg (`-vf fps=1 -q:v 2`). Record each frame's timestamp = clip start + index.
3. **Geotag:** interpolate lat/lon and heading from the GPX track for each frame timestamp.
4. **Snap to block faces:** nearest géobase segment within 15 m (in EPSG:32188). Use heading and the clip's camera side from the manifest to pick the block face: R = the face on the right of travel, L = the face on the left. Keep L frames only on one-way streets (géobase SENS_CIR != 0). Drop frames within about 10 m of intersections.
5. **Blur:** faces and licence plates with EgoBlur (`pip install egoblur`) on frames that will be published. Analysis can run on unblurred frames locally.
6. **Filter:** run Ultralytics YOLO11 (pretrained COCO) on all frames, only as a free pre-filter before Claude. Keep frames with a curbside vehicle, a sign-like object, or a hydrant.
7. **VLM extraction:** send filtered frames (or sign crops) to a vision-language model with a fixed JSON schema. Pilot photos and the validation set run through Claude Code within Arvin's plan. For the full run, first compare a free local model with the Anthropic API on 100 photos, then choose. Ask before spending more than $5. Use batching if available. Cache every response to disk keyed by frame ID so nothing is paid for twice. Schema:
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
8. **Validation (owner TBD, likely partner):** a labelling helper (simple local HTML page or notebook) for hand-labelling stratified frames. 300 drawn, first 100 labelled (cut Nov 18). Compute precision, recall and F1 per class. Kappa dropped (one labeller).
9. **Reconcile (side finding):** compare video sign reads with City sign records within 10 m. Label each as confirmed, missing in data, missing on street, or conflicting.

## Analysis spec (to build in analysis/)

- **Unit:** block face (one side of a street between two intersections) by time window (07 to 11, 11 to 15, 15 to 19, weekdays). Supply and demand cover all 522 faces. Observed behaviour covers the sample only.
- **Block faces:** done, see analysis/blockfaces.py. ID `<street>_<from>_<to>_<N|S|E|W>`. Length in metres.
- **Sample:** done, see analysis/sample.py. Stratified cluster sample, whole blocks (both sides), seed 540. 5 strata: loading (blocks with a loading sign or official loading zone), and for the rest arterial/collector (class 5 to 7) vs local (class 0) x occupied 2025 premises per 100 m above vs at-or-below the median (0.80). Random blocks per stratum: 5 art_high, 5 loc_high, 4 art_low, 4 loc_low, 2 loading. Forced (weight 1): the 12 blocks holding official loading zones with no matching sign. 62 faces: loading 28 (24 forced), art_high 10, loc_high 10, loc_low 8, art_low 6. Weights = blocks in stratum pool / blocks drawn, and sum back to the frame.
- **Supply:** linear metres per block face by use and time window. Uses: paid, free, loading, no stopping, no parking, permit (S3R), reserved (taxi, hotel, police, diplomatic, disabled, EV, moto, bus), implicit no parking (5 m intersection clearance, hydrants, driveways). Parse RPA descriptions into use, hours and days through a rule CSV Arvin reviews. Street cleaning counts only on its days and hours. Permit and reserved space is not available for deliveries but stays as separate categories. Pick one source for paid parking (signs or Places.csv) to avoid double counting. Hydrants and driveways: City data if it exists, otherwise only on sampled faces from video (limitation).
- **Demand:** 2025 commercial premises, occupied, on the curb (mall units use the mall's docks and are counted apart). Rates per SCIAN 3-digit code in analysis/delivery_rates.csv. Low = Sánchez-Díaz (2017) Table 3 weekly rates ÷ 5. High = Holguín-Veras et al. (2011) NYC models (via NCHRP 739 Table 24), constant + rate × assumed staff. Spread over the hour with the Gothenburg cordon profile, Sánchez-Díaz (2017) Table 9, delivery columns only (peak 09 to 10, 18% of the day).
- **Usable curb:** loading and short-term in full, plus a vacant share (15%, placeholder for video occupancy) of paid and free parking. Spaces = metres / 10 m, fractional.
- **Loading model:** Erlang B per hour. The two faces of a street block pool trucks and spaces. Required = fewest spaces that keep P(no space) under 10% at the window's peak hour, split between the two faces by deliveries. 4 combinations: low/high demand × dwell 13.4 min (Kim et al. 2021) / 16.5 min (Schmid et al. 2018). Sensitivity: face alone, 30% loading misuse (placeholder), signed loading only.
- **Mismatch index:** MI = (required − supplied) / max(required, supplied), from −1 (oversupply) to +1 (undersupply). 0 when a face has neither.
- **Model:** done, see analysis/double_parking.py. Unit = street block (both faces), since the forward camera can't tell which curb a truck is on along one-way streets. Count of events, offset log(100 m of curb filmed, summed over days). Poisson with robust errors (NB only as a check, too few events to estimate dispersion). One predictor at a time: MI, P(no space), log2 deliveries a day. Too few events for more covariates. Report rate ratios.
- **Partner's side (support when asked):** collisions within 25 to 30 m of loading activity, bike-lane blocking, sidewalk width samples, curb share for people versus vehicle storage.

## Website spec (to build in docs/ for GitHub Pages)

- MapLibre GL JS, plain HTML/JS, no build step. Single `index.html`, `app.js`, `style.css`, `blockfaces.geojson`.
- Map coloured by mismatch index. Layer toggles: regulations, demand, observed double parking, collisions, bike lanes.
- Sliders: convert N paid spaces per block face to loading, loading time window, dwell time, vehicle length, share of deliveries consolidated to cargo bikes, off-peak delivery share. Recompute MI in the browser.
- Summary panel: block faces in undersupply, loading metres added, parking spaces and estimated revenue lost (rates from Places.csv), estimated change in double parking.
- 3 or 4 preset scenarios.
- Distinctive, clean design. It must not look like a generic template or AI-generated. Works on phone width.
- Data per block face uses our own simple schema. CDS export is optional, only if time allows.

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
- Nov 4: block faces built from géobase double side lines, chained between intersections on the centre lines. ID = <street>_<from>_<to>_<side>. Side is N/S/E/W in Montréal grid terms (grid east = 34 degrees true). from/to run grid west to east or south to north. "split" = where a street splits into two carriageways, "end" = dead end. Median sides of divided streets and junction connectors with no addresses are dropped. Boundary streets keep both sides.
- Nov 4: final scope is "Option 2": stratified cluster sample of about 55 block faces, 3 weekday morning bike passes (08:00 to 11:00). No full inventory ride. No fixed-camera sessions. Delivery dwell time from the literature (10 to 20 min) with sensitivity.
- Nov 4: headline question is where downtown lacks loading space, whether that predicts double parking, and how much curb would fix it. The sign check is a side finding.
- Nov 4: CurbLR dropped. Own simple schema. OMF CDS export optional at the very end.
- Nov 4: YOLO only as a free pre-filter before Claude. 1 frame per second on the passes.
- Nov 4: pilot photos and the validation set run through Claude Code within Arvin's plan. Full run: compare a free local model with the API on 100 photos first. Ask before spending more than $5.
- Nov 4: partner split being rebalanced (partner likely owns website and validation labelling). TBD.
- Nov 4: no riding this week. Test clip before Monday. Pilot Mon Nov 9 midday. Morning passes Tue to Thu Nov 10 to 12, backups Fri Nov 13 and Mon Nov 16.
- Nov 4: supply rules. Street cleaning counts only on its days and hours. S3R permit and reserved spots are not available for deliveries but stay as separate categories.
- Nov 4: paid parking comes from Places.csv and its regulation tables, not signs. Places has every paid space with a position and its paid hours. Signs only carry 11 zone-level "STATIONNEMENT TARIFÉ" plates. Using one source avoids double counting. Time-specific bans on signs (rush hour no stopping, street cleaning) still override paid space in the windows they apply.
- Nov 4: arrow codes (FLECHE_PAN) read for a viewer standing in the street facing the sign: 2 = left, 3 = right, 0 = no arrow (rule runs both ways from the pole), 8 = rare (4 signs), treated as both ways. Inferred from same-rule sign pairs, to be confirmed on Street View.
- Nov 4: fetch_data.py bug fixed. clip_all deleted the whole GeoPackage even with --only. Now it only rebuilds from scratch on a full run.
- Nov 4: sign extents follow the Agence's chaining logic. A rule runs from its pole in the arrow's direction to the next pole with the same rule, or to the end of the face. No arrow = both ways. 76% of sign stretches reach a face end, which may overstate a few zones. Video on sampled faces tests this.
- Nov 4: supply is resolved per weekday half-hour, 07:00 to 19:00, on reference date Nov 10, 2026 (filming week, street cleaning still in force). Strictest rule wins: no_stopping > bus_stop > no_parking > implicit_no_parking > reserved > permit > loading > short_term > paid > free. Metres per window are averages over its half-hours.
- Nov 4: short-term parking (P 15 MIN etc.) counts as usable by delivery drivers but is kept separate from loading.
- Nov 4: paid spaces from Places are 6 m each (median centre spacing 6.1 m). Neighbours up to 9 m apart form one continuous row. Hydrants and intersections get 5 m no parking (Quebec Highway Safety Code). Intersection clearance is applied at both ends of every face, including the 4 dead-end faces.
- Nov 4: 12 of 62 official loading zones (Aires de livraison) have no matching loading sign within 15 m. 3 have no rule code, 1 (SL-JL) has no active sign. Signs stay the source for loading. All 12 faces go into the video sample so the street settles it (side finding for the sign check).
- Nov 4: loading-zone blocks are their own stratum, not all forced in. Forcing all 58 loading blocks (115 faces) broke the sample size. Only the 12 blocks where the official loading list and the signs disagree are forced in.
- Nov 4: two camera loops per morning (right, then left), since 78% of blocks are one-way and left-side curbs can't be filmed on the right riding legally.
- Nov 4: random blocks per stratum cut from 7/7/6/6 to 5/5/4/4. 7/7/6/6 gave 78 faces and about 164 min per morning. 5/5/4/4 gives 62 faces and about 132 min, leaving slack in 08:00 to 11:00. Cost: random strata have 6 to 10 faces, not 12 to 15.
- Nov 4: premises matched to faces by street + civic number + odd/even side, accepted only within 40 m of the closest face on that street (overlapping address ranges sent a few to far blocks). Mall units go to the mall's frontage face. 1,287 occupied of 1,632.
- Nov 4: route graph = full géobase within 300 m of the study area, one-way rules respected for filming. Bidirectional bike tracks and city-street underpasses (class 5 tunnels) are used only for transit. Task order optimised by local search on shortest legal paths. Ride time assumes 10 km/h filming, 14 km/h transit, plus 15% for stops.
- Nov 7: filming by car, not bike. The partner drives. Replaces the two bike loops and the Nov 4 route decision above.
- Nov 7: two mounted iPhones. R on the front passenger window (right curb, all streets). L on the rear driver-side window (left curb, one-way streets only). No hand-held footage. Each phone runs its own GPX logger. One clip per phone per morning, named YYYYMMDD_R/L.MOV and .gpx. The pipeline reads the camera side from the file name and keeps L frames only on one-way streets.
- Nov 7: passes Mon Nov 9, Wed Nov 11, Fri Nov 13, leaving 08:15. Backup Mon Nov 16. A weekend test drive replaces the pilot ride.
- Nov 7: car route = one loop from Peel and Ste-Catherine covering all 62 faces. Graph = géobase within 600 m of the study area (300 m left one-way streets dead-ending at the edge), no pedestrian streets, expressways, ramps, alleys or park roads. Tunnels for transit only. Routing runs on movements (segment to segment), so turn rules apply. Turn bans come from the road sign layer (OBLIG. TOUT DROIT OU A DROITE = no left etc.), matched to an approach only when the pole is clearly on that street (within 15 m, 4 m closer than the cross street, 40 m from a real intersection). Time-limited signs are treated as always in force. 76 of 327 turn signs matched. No U-turns, including left-left through short median crossovers and turns sharper than 125°. The script checks every move is legal. Drive time at 15 to 20 km/h average.
- Nov 7: demand uses two scenarios from the course sources, not NCFRP 37. Low = Sánchez-Díaz (2017) weekly rates ÷ 5. High = Holguín-Veras et al. (2011) NYC models. Offices: 0 (low) and the services rate (high). Hotels use the food services rate. Both flagged for review.
- Nov 10: signed loading zones alone cover only 40 of 522 faces, so loading-only supply sends 85% of trucks to double parking and gives a near-binary map. Usable curb = loading + short-term + 15% of paid and free curb (vacant share, to be replaced by video occupancy). Loading-only stays as a sensitivity test.
- Nov 10: required loading is computed per street block (both faces pooled, drivers cross the street) and split by deliveries. Per-face Erlang made every face with one business need a full 10 m space. Limitation: faces with no stopping all day (east Sainte-Catherine) send trucks to cross streets, which pooling across the street does not capture.
- Nov 10: hourly profile from the Gothenburg cordon count (Sánchez-Díaz 2017 Table 9, delivery columns), spread from half-hour periods to clock hours. Window peak hours: 09, 12, 15. Fractional spaces handled by interpolating Erlang B.
- Nov 13: day 1 (Nov 9) has no GPX log. Day 2 (Nov 11, Remembrance Day closures) followed a different route. Both only have the R camera. Routes may differ every day: the pipeline snaps by GPS, never by planned route order, and the double-parking model uses passes per face as exposure.
- Nov 13: clips keep their original names (IMG_xxxx.MOV). pipeline/clips.csv maps each to date, side (R by default, set L by hand) and part, and pairs it with the GPX that overlaps it in time. This replaces the YYYYMMDD_R.MOV naming rule. Frame folders use the file stem so edits to side don't move them.
- Nov 13: a frame is snapped to the closest block face within 25 m that runs within 35 degrees of travel and lies on the camera's side. GPS gaps are trusted if the phone moved under 15 m across them (trackers skip fixes when stopped). Stopped = under 1 m/s.
- Nov 13: days 1 and 2 were filmed with one iPhone looking forward through the windshield (mounted upside down, 1080p at 60 fps), not a side-window R camera. New camera type F in clips.csv: matched from a point 15 m ahead of the car, counts for the right curb everywhere and for the left curb too on one-way streets. Signs face oncoming traffic, so F reads them well, and double parking ahead is visible.
- Nov 13: the side of the street comes from the two curb lines of the matched street block, not from the GPS position. Downtown GPS error across a narrow street flipped sides before (Drummond).
- Nov 16: work split settled. Partner owns validation labelling (300 frames, cut to 100 on Nov 18) and the safety side. Arvin builds the website.
- Nov 16: project moved from ~/Desktop to ~/cive540-curb. The Desktop syncs to iCloud with Optimize Mac Storage, and macOS offloaded frames, data and git objects whenever the disk ran low, which made reads time out.
- Nov 16: the partner's clips were exported to OneDrive, which replaced the recording time with the export time and cut the right camera from 1080p to 720p. Start times come from the Photos info panel (pipeline/start_times.csv), fine-tuned by sync.py.
- Nov 16: Monday (Nov 9) is not geotagged. No GPX, and Wednesday + Friday already cover 56 of 62 sample faces. Monday frames (forward + left) stay usable for sign reading without a position. (Replaced Nov 17, see below.)
- Nov 17: Monday and the start of Friday (09:29 to 09:44, before the GPX) are geotagged from a route log the partner wrote while watching the L clips: street, entry and exit time, turn. Each turn = the crossing of the two streets in the géobase at the logged time. Between turns the car follows the centre line, with distance shared out by speed predicted from frame-to-frame motion (curve fitted on L clips with GPS). Open ends are walked along the street from the nearest turn. Positions are less exact than GPS (median 13 m along the street vs about 5 m), so these frames carry a source flag later if needed.
- Nov 16: 6 sample faces had construction and were not filmed on any day: Sainte-Catherine Peel-Metcalfe N/S, Stanley Cypress-Sainte-Catherine E/W, Sherbrooke Union-Aylmer N, Sherbrooke Victoria-Robert-Bourassa N. They drop out of the observed sample (limitation). No backup drive. (Nov 17: Monday covers 4 of them. Only Sainte-Catherine Peel-Metcalfe N/S remain unfilmed.)
- Nov 16: Apple GPU (MPS) crashes with torch 2.10 on macOS 14, so YOLO runs on the CPU (15 min for 2,397 frames).
- Nov 16: YOLO filters little downtown (87% of frames have a near vehicle). Dropping repeat frames while stopped does most of the cutting.
- Nov 16: VLM = Claude Haiku 5.5 with the frames 2 s before and after. Matches Sonnet on parked counts and vehicle types at a tenth of the cost. Neither model reads double parking reliably from photos (rare, and a stopped truck looks like one waiting in traffic). So double parking is VLM-assisted, human-verified: the VLM finds candidates (any vehicle outside the curb lane, or any van, box truck or bus), Arvin confirms each in review.html.
- Nov 16: API key in .env (gitignored). Spend so far about $3.
- Nov 18: empty share of paid and free curb set to 30%. The VLM reads 51% empty (CI 44 to 58%) on side-camera frames, but hand checks show it under-calls full curbs, counts BIXI stations and the Sainte-Catherine shared street as curb, and a third of frames mostly show an intersection. So 51% is an upper bound. The worst 50 faces are the same at 15%, 30% and 50%, only totals move. Misuse is not measurable (9 frames on active loading curb) and stays a 30% assumption with sensitivity.
- Nov 18: hand labels cut from 300 to the first 100 frames of validation/sample.csv. Double parking is now human-reviewed, so the VLM's accuracy no longer drives the headline. 100 frames still give per-class precision and recall for the method section, and they are the 100 the VLM test already ran on. They cover all 5 strata and all 3 cameras.
- Nov 18: double parking is reviewed by the partner alone (no second reviewer, no kappa on it, a limitation). Coding rule: half in the lane and half on the sidewalk counts as double parked (flagged partial). Sidewalk only is not double parking (kept as other illegal stop). A bus at its own stop is no. Frames F_1 1039 to 1041 (Nov 11) are one event, coded double parked. Events are merged by time across cameras. API spend now about $3.80.
- Nov 20: double parking modelled per street block, not per face. Events = reviewed positive frames on the same block within 60 s, across cameras. Exposure = share_seen x face length summed over faces and days. With 9 events the model has one predictor at a time. Result: demand predicts double parking, the supply-based shortage (MI, P(no space)) does not. Likely reasons: MI saturates at +1 on any block with demand and no usable curb, however small the demand, and drivers double park even where curb exists (short stops, curb taken by others). Report as is, with demand as the validated part of the model.
- Nov 20: tested trucks with no space per block (a count that grows with demand) in place of the MI ratio. It does not add to deliveries a day (joint model RR 1.03, CI 0.69 to 1.54). Blocks the model says have enough curb still double park (Crescent 82 m usable vs 50 m needed, Guy, Sainte-Catherine Metcalfe-Mansfield), and blocks with no usable curb and moderate demand showed none (De-Maisonneuve Pierce-Guy, Sherbrooke Victoria-Robert-Bourassa). With 9 events the test can't rule out a moderate supply effect. Story: demand is validated, the supply side is not. The website leads with demand and shows the shortage as a model output, not an observed link.
- Nov 20: power check by simulation (9 events, observed exposure, blocks split by MI > 0): a true 2-fold supply effect reaches p < 0.05 only 20% of the time, 3-fold 39%, 4-fold 54%. So the supply result is a non-finding, not evidence of no effect. Deliveries p = 0.010 (high) and 0.003 (low) clear Bonferroni over 4 predictors. 7 of 9 events were on Wednesday, 0 on Monday. Story doc for the partner: https://claude.ai/artifact/BsHqkPHbKtF2zoEswWYuy1. Plan: partner builds the deck from it, Arvin makes the figures, then the website.
- Nov 20: occupied businesses per 100 m of curb (no delivery rates) predicts double parking as well as modelled deliveries, a bit better (RR 2.07 per doubling, CI 1.45 to 2.96, AIC 38.9 vs 43.3, correlation 0.95). So the video validates 'commercial blocks double park more', not the delivery rates or the loading model. Story reframed: the method is the contribution, the result is a pilot. Website slide added to the deck outline.
- Nov 20: website look = planning document (draft A). Dark operations map dropped: it suggests live, precise data we don't have, and it is cramped on phones. Kept from it: short faces drawn thicker, and a 'parking spaces given up' number so every scenario shows its cost.
