# YTBTrendingDashboard — setup (this folder is the git repo)

## 1. Clone

```bash
git clone <url> YTBTrendingDashboard
cd YTBTrendingDashboard
```

## 2. Python (training + `scripts/run_predict_pipeline.py`)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate    # macOS/Linux
pip install -r requirements.txt
```

Put training CSVs in **`ml/data/`** (or keep them next to this repo in a parent folder — the scripts also look for `../stage1_training_data.csv`):

- `ml/data/stage1_training_data.csv`
- `ml/data/stage2_training_data.csv`

Or set environment variables:

- `STAGE1_TRAINING_CSV` — full path to stage 1 CSV
- `STAGE2_TRAINING_CSV` — full path to stage 2 CSV

Train models (writes JSON under **`models/`**):

```bash
python ml/stage_1_xg_boost.py
python ml/stage_2.py
```

## 3. Next.js

```bash
npm install
```

Create **`.env`**:

```env
API_KEY=<YouTube Data API v3 key>
```

Optional:

- `PYTHON_PATH` — Python executable for predictions
- `DASHBOARD_ROOT` — absolute path to this repo if Next’s `cwd` is not this folder

```bash
npm run dev
```

## 4. Predictions

Video Overview calls **`POST /api/trending-predict`**, which runs **`scripts/run_predict_pipeline.py`** with `cwd` = `scripts/`. The API also sets **`PYTHONPATH`** to the **`ml/`** folder so imports work on a clean machine.

### If prediction fails on another computer

1. Run **`npm run dev` from this repo root** (the folder that contains `package.json` and `scripts/`). If you start Next elsewhere, set **`DASHBOARD_ROOT`** to the absolute path of this repo.
2. Install Python deps: **`pip install -r requirements.txt`** (same machine where Node runs, or set **`PYTHON_PATH`** to that interpreter).
3. Ensure **`models/*.json`** exists (train with `python ml/stage_1_xg_boost.py` etc., or copy the four files).
4. On Windows, if `python` is missing, install Python 3.10+ from python.org and tick “Add to PATH”, or set **`PYTHON_PATH`** to the full path of `python.exe`.
5. Read the error text on Video Overview — it now includes **detail** (stderr / Python message) and a short **hint**.
