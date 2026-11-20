# Curb mismatch in downtown Montréal

CIVE 540 term project, McGill, Fall 2026. Where does downtown Montréal lack loading space, does that predict double parking, and how much curb would fix it?

Study area: Sherbrooke St W (north), René-Lévesque Blvd W (south), Atwater Ave (west), De Bleury St (east).

Work in progress. Full documentation, reproduction steps and accuracy results will be added when the analysis is done.

## Pipeline so far

```
python fetch_data.py            # City open data, clipped to the study area
python analysis/blockfaces.py   # 522 block faces
python analysis/supply.py       # curb rules per face and time window
python analysis/premises.py     # 2025 businesses matched to faces
python analysis/sample.py       # 62 faces sampled for video
python analysis/route.py        # car loop for filming
python analysis/demand.py       # deliveries per face, low and high scenarios
python analysis/loading.py      # Erlang loading model and mismatch index
```

Video (three weekday morning drives, Nov 9, 11 and 13, raw video and frames are not in the repo):

```
python pipeline/ingest.py           # clips manifest (pipeline/clips.csv)
python pipeline/frames.py           # 1 frame per second
python pipeline/anchors.py          # GPS track for clips with no GPX, from a hand-written route log
python pipeline/sync.py             # clock offset per clip, from video motion vs GPS speed
python pipeline/geotag.py           # position and block face per frame
python pipeline/detect.py           # YOLO11 pre-filter
python pipeline/vlm.py haiku send --context   # Claude Haiku 5.5 reads the curb in each frame
python pipeline/review_list.py --wide         # double-parking candidates for human review
python analysis/occupancy.py        # how full the parked curb is, from the side cameras
python analysis/double_parking.py   # reviewed frames -> events per block, and the test of the loading model
```

## To validate with field notes

The loading model (`analysis/loading.py`) is built on literature values and two shares that the video was meant to measure. Status:

- [x] `VACANT_SHARE`, share of paid and free parking that is empty. Was a 15% placeholder. The video reads 51% (95% CI 44 to 58%), but hand checks show that is an upper bound (the model under-calls full curbs and counts bike-share stations as empty). Now 30%, with 15 to 50% as the range. The 50 worst faces are the same across that range.
- [x] `MISUSE_SHARE`, share of loading curb taken by other vehicles. Could not be measured: only 9 frames fell on loading curb while the rule was in force. Stays a 30% assumption, tested in the sensitivity runs.
- [ ] Check dwell times (13.4 and 16.5 min from Kim et al. 2021 and Schmid et al. 2018). Not observable from drive-by video at 1 frame per second. Literature values with sensitivity.
- [x] Compare predicted P(no space) with observed double parking per km of curb. Done per street block (9 events on 31 blocks). Delivery demand predicts double parking (rate ratio 1.5 per doubling, 95% CI 1.1 to 2.1). The shortage index and P(no space) do not (rate ratio 1.04 per +0.1, CI 0.96 to 1.12). Blocks with 9 or more deliveries a day: 1.39 events per km, others 0.22. Trucks that find no space (a count that grows with demand) also point the right way (rate ratio 1.3 per doubling, CI 0.9 to 1.9) but add nothing once deliveries are known. With 9 events, a moderate supply effect can't be ruled out.
- [ ] Check faces with no stopping all day (east Sainte-Catherine). The model assumes trucks there use the opposite side, but they probably use the cross streets.
- [x] Confirm the flagged rates in `analysis/delivery_rates.csv` (hotels, offices) and leaving shopping-centre units off the curb. Confirmed Nov 18.

## Video status

- [x] Three drives filmed. Monday had no GPS log and is positioned from a route log (median error 13 m along the street, checked on clips that have GPS).
- [x] 60 of 62 sampled faces filmed at least once. Sainte-Catherine between Peel and Metcalfe was never filmed.
- [x] Double-parking review, batch 1 (318 frames, Wednesday and Friday).
- [x] Double-parking review, batch 2 (213 frames, Monday and early Friday).
- [ ] Hand labels for the first 100 frames of the validation sample, to score the vision model.
- [x] Double-parking events per street block, and the model (`analysis/double_parking.py`).

## Data

City of Montréal open data (CC BY 4.0). Paid parking data from the Agence de mobilité durable (CC BY 4.0).
