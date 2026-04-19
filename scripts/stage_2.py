# ======================================================
# Stage 2: Post-trend daily view-rate trajectory (3-class)
# Train + tune + save. Use predict_stage2_post_trend() at inference.
# ======================================================
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent  # YTBTrendingDashboard root
MODELS_DIR = _ROOT / "models"
DEFAULT_STAGE2_MODEL_PATH = MODELS_DIR / "stage2_xgb.json"
DEFAULT_STAGE2_META_PATH = MODELS_DIR / "stage2_meta.json"
DEFAULT_STAGE2_TRAINING_CSV = _ROOT / "stage2_training_data.csv"

CAT_TREND_SPEED: Dict[int, Dict[str, float]] = {
    25: {"p10": 2.338, "p50": 2.864, "p90": 6.510},
    28: {"p10": 3.017, "p50": 3.643, "p90": 8.310},
    2: {"p10": 3.179, "p50": 4.077, "p90": 7.498},
    27: {"p10": 3.194, "p50": 4.504, "p90": 9.825},
    23: {"p10": 3.304, "p50": 4.564, "p90": 9.220},
    17: {"p10": 3.211, "p50": 4.719, "p90": 8.776},
    19: {"p10": 3.408, "p50": 5.064, "p90": 10.074},
    15: {"p10": 3.822, "p50": 5.086, "p90": 9.307},
    29: {"p10": 3.320, "p50": 5.456, "p90": 10.775},
    22: {"p10": 3.962, "p50": 5.561, "p90": 10.211},
    26: {"p10": 4.119, "p50": 6.178, "p90": 10.899},
    24: {"p10": 4.906, "p50": 11.010, "p90": 20.475},
    10: {"p10": 6.029, "p50": 13.873, "p90": 23.914},
    20: {"p10": 5.949, "p50": 13.906, "p90": 24.299},
    1: {"p10": 6.110, "p50": 13.943, "p90": 24.016},
}

EVERGREEN_WORDS = ["how", "tutorial", "guide", "tips", "review", "best", "top", "learn", "explained"]
TIMELY_WORDS = ["breaking", "live", "update", "news", "today", "tonight", "new", "just", "official"]


def hour_bucket(h: int) -> int:
    if 6 <= h < 12:
        return 0
    if 12 <= h < 17:
        return 1
    if 17 <= h < 21:
        return 2
    return 3


def _cat_speed_frame() -> pd.DataFrame:
    cat_speed_df = pd.DataFrame(CAT_TREND_SPEED).T
    cat_speed_df.columns = ["cat_trend_p10", "cat_trend_p50", "cat_trend_p90"]
    cat_speed_df.index.name = "category_id"
    cat_speed_df = cat_speed_df.reset_index()
    cat_speed_df["category_id"] = cat_speed_df["category_id"].astype(int)
    cat_speed_df["cat_trend_spread"] = (
        cat_speed_df["cat_trend_p90"] - cat_speed_df["cat_trend_p10"]
    )
    return cat_speed_df


def _merge_cat_speed(df: pd.DataFrame) -> pd.DataFrame:
    cat_speed_df = _cat_speed_frame()
    out = df.copy()
    out["category_id"] = pd.to_numeric(out["category_id"], errors="coerce")
    drop_cols = [
        c
        for c in ["cat_trend_p10", "cat_trend_p50", "cat_trend_p90", "cat_trend_spread"]
        if c in out.columns
    ]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return out.merge(cat_speed_df, on="category_id", how="left")


def prepare_stage2_frame(
    df: pd.DataFrame,
    *,
    fill_values: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
    """Engineer Stage-2 features (trending-era snapshot schema)."""
    df = df.copy()
    df["publish_time"] = pd.to_datetime(df["publish_time"], utc=True, errors="coerce")

    for col in ["views", "likes", "comments"]:
        if col in df.columns:
            med = df[col].median()
            if pd.isna(med):
                med = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(med)

    if "days_since_publish" in df.columns:
        dsp = pd.to_numeric(df["days_since_publish"], errors="coerce").fillna(0.0)
        df["days_since_publish"] = dsp.clip(lower=0.0)

    df["likes_per_view"] = df["likes"] / (df["views"] + 1)
    df["comments_per_view"] = df["comments"] / (df["views"] + 1)
    df["like_comment_ratio"] = df["likes"] / (df["comments"] + 1)

    df["views_per_day"] = df["views"] / (df["days_since_publish"] + 1)
    df["log_views"] = np.log1p(df["views"])
    df["log_vpd"] = np.log1p(df["views_per_day"])

    df["publish_hour"] = df["publish_time"].dt.hour.fillna(12).astype(int)
    df["publish_dayofweek"] = df["publish_time"].dt.dayofweek.fillna(0).astype(int)
    df["publish_period"] = df["publish_hour"].map(hour_bucket)
    df["is_weekend"] = (df["publish_dayofweek"] >= 5).astype(int)

    title_features: List[str] = []
    if "title" in df.columns:
        t = df["title"].astype(str)
        df["title_length"] = t.str.len()
        df["title_word_count"] = t.str.split().str.len()
        df["title_caps_ratio"] = t.apply(
            lambda s: sum(1 for c in s if c.isupper()) / max(len(s), 1)
        )
        df["title_has_exclaim"] = t.str.contains("!", regex=False).astype(int)
        df["title_has_question"] = t.str.contains(r"\?", regex=True).astype(int)
        df["title_has_number"] = t.str.contains(r"\d", regex=True).astype(int)
        tl = t.str.lower()

        def _count_words(text: str, words: List[str]) -> int:
            parts = text.split()
            return sum(1 for w in words if w in parts)

        df["title_evergreen_count"] = tl.apply(lambda x: _count_words(x, EVERGREEN_WORDS))
        df["title_timely_count"] = tl.apply(lambda x: _count_words(x, TIMELY_WORDS))
        title_features = [
            "title_length",
            "title_word_count",
            "title_caps_ratio",
            "title_has_exclaim",
            "title_has_question",
            "title_has_number",
            "title_evergreen_count",
            "title_timely_count",
        ]

    df = _merge_cat_speed(df)
    cat_cols = ["cat_trend_p10", "cat_trend_p50", "cat_trend_p90", "cat_trend_spread"]
    for col in cat_cols:
        if fill_values and col in fill_values:
            df[col] = df[col].fillna(fill_values[col])
        else:
            df[col] = df[col].fillna(df[col].median())

    cat_speed_features = ["cat_trend_p10", "cat_trend_p50", "cat_trend_p90", "cat_trend_spread"]

    features = (
        [
            "category_id",
            "days_since_publish",
            "likes_per_view",
            "comments_per_view",
            "like_comment_ratio",
            "views_per_day",
            "log_views",
            "log_vpd",
            "publish_hour",
            "publish_dayofweek",
            "publish_period",
            "is_weekend",
        ]
        + title_features
        + cat_speed_features
    )

    fill_out: Dict[str, float] = {}
    for col in cat_cols:
        if col in df.columns:
            fill_out[col] = float(df[col].median())

    return df, features, fill_out


def train_stage2_and_save(
    csv_path: Path | str = DEFAULT_STAGE2_TRAINING_CSV,
    model_path: Path | str = DEFAULT_STAGE2_MODEL_PATH,
    meta_path: Path | str = DEFAULT_STAGE2_META_PATH,
) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    csv_path = Path(csv_path)
    model_path = Path(model_path)
    meta_path = Path(meta_path)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Stage 2: load CSV ===")
    df_raw = pd.read_csv(csv_path)
    target = "vpd_class"
    if target not in df_raw.columns:
        raise ValueError(f"Expected '{target}' in {csv_path}")

    df, features, fill_values = prepare_stage2_frame(df_raw)
    X = df[features].astype(float)
    y = df[target].astype(int)

    print(f"Samples: {len(df)}, features: {len(features)}")

    print("\n=== Hyperparameter tuning ===")
    param_dist = {
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0, 0.5, 1, 2],
        "reg_alpha": [0, 0.1, 0.5, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0],
    }

    base_model = xgb.XGBClassifier(
        n_estimators=500,
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=50,
        scoring="roc_auc_ovr",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X, y)
    best_params = dict(search.best_params_)
    print(f"Best CV ROC-AUC (OVR): {search.best_score_:.4f}")

    print("\n=== Final fit (early stopping) ===")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    model = xgb.XGBClassifier(
        n_estimators=500,
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        verbosity=0,
        **best_params,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    best_it = int(model.best_iteration) if model.best_iteration is not None else 500
    best_it = max(best_it, 1)
    print(f"best_iteration: {best_it}")

    print("\n=== Refit on full data ===")
    final_model = xgb.XGBClassifier(
        n_estimators=best_it,
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0,
        **best_params,
    )
    final_model.fit(X, y, verbose=False)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(model_path))
    print(f"Saved model: {model_path}")

    class_labels = [
        "Fading — low sustained views/day after trending ends",
        "Steady — moderate post-trend velocity",
        "Thriving — high post-trend velocity",
    ]

    imp = final_model.feature_importances_
    top_idx = np.argsort(-imp)[:5]
    top_importance = [
        {"feature": features[i], "importance": float(imp[i])} for i in top_idx
    ]

    meta: Dict[str, Any] = {
        "features": features,
        "fill_values": {k: float(v) for k, v in fill_values.items()},
        "best_params": best_params,
        "best_iteration": best_it,
        "num_class": 3,
        "class_labels": class_labels,
        "feature_importance_top5": top_importance,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved metadata: {meta_path}")

    return final_model, meta


# --- inference ---
_s2_model: Optional[xgb.XGBClassifier] = None
_s2_meta: Optional[Dict[str, Any]] = None
_s2_paths: Optional[Tuple[str, str]] = None


def _load_stage2(model_path: Path, meta_path: Path) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    global _s2_model, _s2_meta, _s2_paths
    key = (str(model_path), str(meta_path))
    if _s2_model is not None and _s2_paths == key:
        return _s2_model, _s2_meta  # type: ignore[return-value]

    if not meta_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(
            f"Stage 2 model files missing ({model_path}, {meta_path}). Run train_stage2_and_save()."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    clf = xgb.XGBClassifier()
    clf.load_model(str(model_path))
    _s2_model, _s2_meta, _s2_paths = clf, meta, key
    return clf, meta


def _days_since_publish(record: Dict[str, Any]) -> float:
    if "days_since_publish" in record and record["days_since_publish"] is not None:
        return max(0.0, float(record["days_since_publish"]))
    pt = record.get("publish_time") or record.get("publishedAt")
    if not pt:
        return 0.0
    pub = pd.to_datetime(pt, utc=True, errors="coerce")
    if pd.isna(pub):
        return 0.0
    now = pd.Timestamp.now(tz="UTC")
    return max(0.0, (now - pub).total_seconds() / 86400.0)


def predict_stage2_post_trend(
    record: Dict[str, Any],
    *,
    model_path: Path | str | None = None,
    meta_path: Path | str | None = None,
) -> Dict[str, Any]:
    """
    Predict post-trend trajectory class (0=fading, 1=steady, 2=thriving).

    Record should include views/likes/comments (video), publish_time, category_id, title.
    """
    model_path = Path(model_path or DEFAULT_STAGE2_MODEL_PATH)
    meta_path = Path(meta_path or DEFAULT_STAGE2_META_PATH)
    model, meta = _load_stage2(model_path, meta_path)
    features: List[str] = meta["features"]
    fill_values: Dict[str, float] = meta.get("fill_values", {})

    row: Dict[str, Any] = dict(record)
    if "category_id" not in row and "categoryId" in row:
        row["category_id"] = row["categoryId"]
    if "publish_time" not in row and "publishedAt" in row:
        row["publish_time"] = row["publishedAt"]
    if "views" not in row and "viewCount" in row:
        row["views"] = row["viewCount"]
    if "likes" not in row and "likeCount" in row:
        row["likes"] = row["likeCount"]
    if "comments" not in row and "commentCount" in row:
        row["comments"] = row["commentCount"]
    row.setdefault("title", "")
    row.setdefault("views", 0)
    row.setdefault("likes", 0)
    row.setdefault("comments", 0)
    if row.get("category_id") is None or (isinstance(row.get("category_id"), str) and row["category_id"] == ""):
        row["category_id"] = 24
    row["days_since_publish"] = _days_since_publish(row)

    df = pd.DataFrame([row])
    df, _, _ = prepare_stage2_frame(df, fill_values=fill_values)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Stage 2 frame missing columns: {missing}")

    X = df[features].astype(float)
    cls = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0].tolist()
    labels: List[str] = meta.get(
        "class_labels",
        ["Fading", "Steady", "Thriving"],
    )
    return {
        "vpd_class": cls,
        "vpd_class_label": labels[cls] if cls < len(labels) else str(cls),
        "class_probabilities": [round(float(p), 6) for p in proba],
        "predicted_class_confidence": round(float(max(proba)), 6),
    }


def main() -> None:
    train_stage2_and_save()


if __name__ == "__main__":
    main()
