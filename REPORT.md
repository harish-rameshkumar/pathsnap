# PathSnap – Technical Report

## 1. Problem and approach
Match noisy 1 Hz GPS fixes to a road network. We generate candidate arcs per fix, score each candidate with a
supervised classifier (P[arc is the reference arc]), then decode a coherent route with an HMM/Viterbi over candidates.

## 2. Data exploration (`results/dataset_inventory.csv`, `results/figures/dataset_overview.png`)
- 80 train + 10 student trips, each with `.track` (lon, lat, t), `.nodes`, `.arcs` (row = arc id), `.route` (arc ids).
  **Every trip has its own small road network, anywhere in the world** (lon/lat spread globally) – so features are computed in a per-trip local frame.
- Projection: per-trip equirectangular plane centred on the trip (metres). Chosen over UTM because extents are city-scale and it is vectorised; coordinate order (lon, lat) verified against plausible locations.
- `.route` is a connected arc chain (to-node of arc i = from-node of arc i+1) with no per-fix labels; GPS lies within ~5–10 m median of it.
- Steps are nominally 1 s but some trips have large gaps (max 153,064 s in training; student max 210 s). 
- **Data-quality findings:** training trips 33 and 54 have reference routes inconsistent with their GPS (66% / 90% of fixes have no true arc among candidates) – excluded from fitting, retained in the inventory. Trips 2, 72, 61, 43, 66 show moderate mismatch (5–13%), kept. Reference routes are treated as reference, not verified truth.
- Per-fix labels: monotone DP alignment of the track to the route (`pathsnap/labels.py`) – an assumption that affects all metrics.

## 3. Candidates and features
Arcs within 80 m, top 12 by true point-to-segment distance. Candidate miss rate (true arc absent): student-test 0.2% overall (<1% each). 19 features: distance, relative/ratio distance, rank, #candidates, projection fraction, arc length, 5-point smoothed distance (+relative), heading agreement (2 windows), speed, time gap, connectivity support to candidates at t±1, same-arc presence at t±1, node degrees. No feature uses labels.

## 4. Model comparison (`results/model_comparison.csv`, `figures/model_comparison.png`)
9 models, 7 families, trained on a 25% observation subsample of the 80 training trips (392,657 rows, 11.6% positive), 5-fold GroupKFold by trip, then evaluated on the 10 student trips (seed 42).

| Model | CV log loss | Student top-1 | F1 | Log loss | Continuity | Fit s |
|---|---|---|---|---|---|---|
| LogisticRegression | 0.0926 | 0.8746 | 0.852 | 0.089 | 0.764 | 1.5 |
| GaussianNB | 0.7762 | 0.8516 | 0.664 | 0.851 | 0.743 | 0.0 |
| KNN (k=25, dist.) | 0.1268 | 0.8775 | 0.868 | 0.116 | 0.730 | 0.1 (inference 49 s) |
| DecisionTree (d=10) | 0.0828 | 0.8919 | 0.884 | 0.070 | 0.804 | 3.8 |
| RandomForest (200) | 0.0633 | 0.8949 | 0.890 | 0.060 | 0.819 | 44.3 |
| ExtraTrees (200) | 0.0661 | 0.8846 | 0.880 | 0.065 | 0.773 | 11.3 |
| HistGradientBoosting | 0.0642 | 0.8946 | 0.890 | 0.066 | 0.801 | 2.5 |
| LightGBM (untuned) | 0.0729 | 0.8924 | 0.889 | 0.075 | 0.752 | 2.1 |
| MLP (64,32) | 0.0728 | 0.8891 | 0.882 | 0.068 | 0.794 | 28.4 |

Primary selection metric: grouped-CV **log loss** (the HMM multiplies probabilities, so calibration matters more than top-1), with student top-1 and speed as tie-breakers. The top five tree/boosting models are within ~1 point of top-1 (differences likely within noise of 10 trips). Boosting was chosen for tuning: fast fit/inference, good probabilities, large hyperparameter space. RandomForest had the best untuned log loss but 20× the fit time. Note: continuity of classifier-only predictions is low (0.73–0.82) for all models – the motivation for the HMM.

## 5. Bayesian optimisation (`results/bayes_opt.json`, `figures/bayes_opt.png`)
Optuna TPE (8 random start trials), 40 trials, 5-fold GroupKFold on training trips, objective = CV log loss. Space: n_estimators 100–600, learning_rate 0.01–0.3 (log), num_leaves 8–128 (log), min_child_samples 10–200, subsample, colsample_bytree 0.5–1, reg_lambda 1e-3–30. Best: n_estimators 500, lr 0.0201, leaves 53, min_child 94, subsample 0.83, colsample 0.53, λ 5.37; CV log loss 0.0602 (untuned 0.0729). Cost: 1095 s. Gains plateaued after ~20 trials. Refit on **all** training rows (20 s).

## 6. HMM / Viterbi (`pathsnap/hmm.py`)
- **States:** candidate arcs of each fix (arc id preserved per row). **Emission:** α·log of classifier probability renormalised per fix.
- **Transition:** Newson–Krumm: −|d_net − d_gps|/β, where d_net is along-network distance between projections (same arc: fraction difference; otherwise remaining length + bounded Dijkstra + offset). Unreachable pairs: flat penalty (decoding always exists); U-turns extra penalty; large backward move on same arc invalid.
- **Gaps:** time gap > `gap_s` restarts the chain. **No candidates:** fix inherits the neighbouring decoded arc.
- **Tuning:** Optuna on 24 training trips using out-of-fold classifier probabilities (no leakage); objective pointwise accuracy. Best: α=1.05, β=6.97 m, invalid_pen=21.2, uturn_pen=6.6, gap_s=120 (115 s).

## 7. Results on the 10 student-test trips (mean over trips)
| | Pointwise acc. | Route continuity | Route P / R / F1 |
|---|---|---|---|
| Untuned LGBM, classifier only | 0.9085 | 0.799 | 0.896 / 0.976 / 0.934 |
| Untuned LGBM + HMM | 0.9306 | 0.886 | 0.951 / 0.989 / 0.970 |
| Tuned LGBM, classifier only | 0.9118 | 0.823 | 0.913 / 0.977 / 0.943 |
| **Tuned LGBM + HMM (final)** | **0.9330** | **0.886** | 0.951 / 0.989 / 0.969 |

The HMM adds ~2 points pointwise accuracy and ~6 points continuity; tuning the classifier adds only ~0.3 point (probably within noise) – the HMM does most of the work. Per trip, the HMM lowers accuracy on trips 00000040 and 00000091 (see `figures/clf_vs_hmm.png`); weakest trip is 00000077 (0.845, continuity 0.77). Remaining continuity <1 stems from shortest-path jumps across GPS gaps and from alignment/label noise.

## 8. Limitations
Labels depend on my route-to-GPS alignment; reference routes have mismatches (trips 33, 54). Decoder tuned for pointwise accuracy only. Student-test (10 trips) is small; the evaluator set was never used for any choice. Continuity counts arc changes that are directly adjacent only.

## 9. Deployment
FastAPI (`/trips`, `/predict`) + Leaflet UI; see README. All 10 student trips ran through the API end to end (0.1–3 s each).
