# PathSnap – GPS map matching (classifier + HMM/Viterbi)

Pipeline: raw GPS → candidate road arcs (≤12 within 80 m) → 19 features → LightGBM candidate probability
→ HMM/Viterbi over candidates with road-network transitions → predicted arc per GPS point.

## Layout
- `pathsnap/` library: `geo` (loading, local metric projection), `labels` (route alignment, training/eval only), `candidates`, `hmm`, `pipeline`, `metrics`
- `scripts/` experiments (run in order): `00_build_dataset`, `01_compare_models`, `02_tune_bayes`, `03_hmm_and_final`, `04_explore`, `05_figures`
- `api/main.py` FastAPI backend (Laptop 1) · `ui/index.html` UI (Laptop 2)
- `artifacts/model.joblib`, `artifacts/hmm_config.json` inference assets (`dataset.pkl` is a regenerable cache, git-ignored)
- `results/` metrics, logs, figures · `REPORT.md` technical report

## Setup
```bash
pip install -r requirements.txt
export PATHSNAP_DATA="/path/to/Student folder"      # contains train/ and student_test/
cd scripts && python 00_build_dataset.py "$PATHSNAP_DATA" && python 01_compare_models.py && python 02_tune_bayes.py 40 && python 03_hmm_and_final.py
```
(`00_build_dataset.py` must be run from the project root: `python scripts/00_build_dataset.py "$PATHSNAP_DATA"`.)

## Final evaluation
Laptop 1 (API; folder holding the 10 evaluator trips, same per-trip layout incl. `.route` for server-side scoring):
```bash
EVAL_DIR=/path/to/evaluator_test uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Laptop 2 (UI): `cd ui && python -m http.server 8080`, open `http://localhost:8080`, set API URL to `http://<laptop1-ip>:8000`, click *Load evaluation trips*.
The API log (terminal on Laptop 1) shows each `POST /predict`. The reference route is read only inside `pathsnap.pipeline.evaluate`; the prediction path never touches `.route`, and the UI only receives predictions and metrics.
