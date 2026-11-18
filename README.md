# PathSnap – GPS map matching (classifier + HMM/Viterbi)

Pipeline: raw GPS → candidate road arcs (≤12 within 80 m) → 19 features → LightGBM candidate probability
→ HMM/Viterbi over candidates with road-network transitions → predicted arc per GPS point.

## Layout
- `pathsnap/` library: `geo` (loading, local metric projection), `labels` (route alignment, training/eval only), `candidates`, `hmm`, `pipeline`, `metrics`
- `scripts/` experiments (run in order): `00_build_dataset`, `01_compare_models`, `02_tune_bayes`, `03_hmm_and_final`, `04_explore`, `05_figures`
- `api/main.py` FastAPI backend (Laptop 1) · `ui/index.html` UI (Laptop 2)
- `artifacts/model.joblib`, `artifacts/hmm_config.json` inference assets (`dataset.pkl` is a regenerable cache, git-ignored)
- `results/` metrics, logs, figures · `REPORT.md` technical report

## Setup (run everything from the project root)
```bash
pip install -r requirements.txt
export PATHSNAP_DATA="/path/to/Student folder"      # must contain train/ and student_test/
python scripts/00_build_dataset.py "$PATHSNAP_DATA"  # builds artifacts/dataset.pkl (cache, ~2 min)
python scripts/01_compare_models.py                  # 9-model comparison  (~10 min)
python scripts/02_tune_bayes.py 40                   # Optuna tuning       (~18 min)
python scripts/03_hmm_and_final.py                   # HMM tuning, final model, artifacts (~5 min)
python scripts/04_explore.py && python scripts/05_figures.py && python scripts/06_visualise.py   # report figures
```
Scripts 01-06 read `PATHSNAP_DATA` (default in `scripts/common.py` is a local path - set the variable).
The API needs only `artifacts/model.joblib` and `artifacts/hmm_config.json`, both committed, so steps above are not required to run it.

## Final evaluation
Laptop 1 (API; folder holding the 10 evaluator trips, same per-trip layout incl. `.route` for server-side scoring):
```bash
EVAL_DIR=/path/to/evaluator_test uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Laptop 2 (UI): `cd ui && python -m http.server 8080`, open `http://localhost:8080`, set API URL to `http://<laptop1-ip>:8000`, click *Load evaluation trips*.
The API log (terminal on Laptop 1) shows each `POST /predict`. The reference route is read only inside `pathsnap.pipeline.evaluate`; the prediction path never touches `.route`, and the UI only receives predictions and metrics.
