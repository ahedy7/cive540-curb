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

## To validate with field notes

The loading model (`analysis/loading.py`) is built on literature values and two placeholders. **Come back and validate this step once the video and field notes are processed:**

- [ ] Replace `VACANT_SHARE` (15%, share of paid and free parking that is empty) with occupancy observed on the sampled faces.
- [ ] Replace `MISUSE_SHARE` (30%, share of loading curb taken by other vehicles) with observed loading-zone misuse.
- [ ] Check dwell times (13.4 and 16.5 min from Kim et al. 2021 and Schmid et al. 2018) against any dwell we can observe.
- [ ] Compare predicted P(no space) per face with observed double parking per 100 m. This is the key test that the model means something.
- [ ] Check faces with no stopping all day (east Sainte-Catherine). The model assumes trucks there use the opposite side, but they probably use the cross streets.
- [ ] Confirm the flagged rates in `analysis/delivery_rates.csv` (hotels, offices) and leaving shopping-centre units off the curb.

## Data

City of Montréal open data (CC BY 4.0). Paid parking data from the Agence de mobilité durable (CC BY 4.0).
