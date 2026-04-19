"""Read JSON from stdin (video + channel payload), print predict_trending_pipeline result as JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Repo root = YTBTrendingDashboard/ ; ML modules live in ml/
_DASH = Path(__file__).resolve().parent.parent
_ML = _DASH / "ml"
if str(_ML) not in sys.path:
    sys.path.insert(0, str(_ML))

if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON: {e}"}), flush=True)
        sys.exit(1)

    try:
        from stage_1_xg_boost import predict_trending_pipeline

        out = predict_trending_pipeline(payload)
        print(json.dumps(out), flush=True)
    except Exception as e:
        print(json.dumps({"error": str(e)}), flush=True)
        sys.exit(1)
