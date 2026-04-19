# ======================================================
# YouTube Trending Video Classifier — v3 (fully leak-free)
# Train + hyperparameter tuning + save model.
# Use predict_stage1_trending() for inference (e.g. from API / UI).
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

# --- Paths (relative to this file) ---
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent  # YTBTrendingDashboard root
MODELS_DIR = _ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "stage1_xgb.json"
DEFAULT_META_PATH = MODELS_DIR / "stage1_meta.json"
DEFAULT_TRAINING_CSV = _ROOT / "stage1_training_data.csv"
DEFAULT_STAGE2_MODEL_PATH = MODELS_DIR / "stage2_xgb.json"
DEFAULT_STAGE2_META_PATH = MODELS_DIR / "stage2_meta.json"

# Client-facing copy for top global feature importances (Stage 1 trending model)
FEATURE_HINTS: Dict[str, str] = {
    "category_id": "Category matters: some niches trend faster than others.",
    "publish_hour": "Upload hour affects early exposure and competition.",
    "publish_dayofweek": "Day of week shifts audience availability and browse patterns.",
    "publish_period": "Time-of-day bucket (morning/afternoon/evening/night) shapes launch dynamics.",
    "is_weekend": "Weekend vs weekday posting changes competition and watch time.",
    "title_length": "Title length changes click curiosity and clarity in browse surfaces.",
    "title_word_count": "Word count signals specificity vs generic clickbait.",
    "title_caps_ratio": "Heavy caps can read as loud or spammy—balance matters for CTR.",
    "title_has_exclaim": "Exclamation can lift urgency but may reduce trust if overused.",
    "title_has_question": "Questions can improve curiosity-driven clicks when matched to intent.",
    "title_has_number": "Numbers/lists often improve skim-ability and perceived concreteness.",
    "title_has_emoji": "Emoji can stand out in feeds but impact varies by category and brand tone.",
    "tag_count": "More tags can widen metadata relevance if they stay honest and on-topic.",
    "desc_length": "Description length correlates with depth of packaging and links/context.",
    "desc_has_links": "Outbound links change how the video is framed as resource vs pure entertainment.",
    "channel_views": "Total channel views proxy overall reach and brand familiarity.",
    "channel_subscribers": "Subscriber scale affects early velocity and recommendation ceilings.",
    "channel_videos": "Catalog size changes per-video attention and posting cadence effects.",
    "subs_per_video": "Subs per upload hints at channel efficiency vs breadth of output.",
    "views_per_video": "Average views per upload signals typical performance level.",
    "log_subscribers": "Log-scaled audience size captures diminishing returns at huge scale.",
    "log_channel_views": "Log-scaled channel views stabilizes heavy-tailed channel popularity.",
    "cat_trend_p10": "Category-specific fast-trending timing (lower tail) vs your publish context.",
    "cat_trend_p50": "Typical category time-to-trend dynamics vs your timing signals.",
    "cat_trend_p90": "Slow-trend tail for the category—helps calibrate patience vs burst potential.",
    "cat_trend_spread": "How variable trending speed is within the category (risk/volatility proxy).",
}

# Category trending speed features (same as training)
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


def _build_feature_groups(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    channel_features: List[str] = []
    for col in ["channel_views", "channel_subscribers", "channel_videos"]:
        if col in df.columns:
            channel_features.append(col)

    if len(channel_features) == 3:
        df["subs_per_video"] = df["channel_subscribers"] / (df["channel_videos"] + 1)
        df["views_per_video"] = df["channel_views"] / (df["channel_videos"] + 1)
        df["log_subscribers"] = np.log1p(df["channel_subscribers"])
        df["log_channel_views"] = np.log1p(df["channel_views"])
        channel_features += [
            "subs_per_video",
            "views_per_video",
            "log_subscribers",
            "log_channel_views",
        ]

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
        df["title_has_emoji"] = t.apply(
            lambda s: int(bool(re.search(r"[^\w\s,.\-!?;:\'\"()#@&/\\]", s)))
        )
        title_features = [
            "title_length",
            "title_word_count",
            "title_caps_ratio",
            "title_has_exclaim",
            "title_has_question",
            "title_has_number",
            "title_has_emoji",
        ]

    tag_features: List[str] = []
    if "tags" in df.columns:
        df["tag_count"] = df["tags"].astype(str).apply(
            lambda x: len(x.split("|")) if x not in ("[none]", "nan") else 0
        )
        tag_features = ["tag_count"]

    desc_features: List[str] = []
    if "description" in df.columns:
        d = df["description"].astype(str)
        df["desc_length"] = d.apply(lambda x: len(x) if x != "nan" else 0)
        df["desc_has_links"] = d.str.contains(r"http[s]?://", regex=True, na=False).astype(int)
        desc_features = ["desc_length", "desc_has_links"]

    cat_speed_features = ["cat_trend_p10", "cat_trend_p50", "cat_trend_p90", "cat_trend_spread"]
    return channel_features, title_features, tag_features, desc_features, cat_speed_features


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


def prepare_frame(
    df: pd.DataFrame,
    *,
    fill_values: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
    """Clean + engineer features. If fill_values is set (inference), use it for NA fills; else use column medians."""
    df = df.copy()
    df = df.dropna(subset=["channel_title"]) if "channel_title" in df.columns else df
    if "days_since_publish" in df.columns:
        df = df[df["days_since_publish"] <= 60]

    for col in ["views", "likes", "comments"]:
        if col in df.columns:
            med = df[col].median()
            if pd.isna(med):
                med = 0.0
            df[col] = df[col].fillna(med)

    channel_cols = ["channel_views", "channel_subscribers", "channel_videos"]
    for col in channel_cols:
        if col in df.columns:
            if fill_values and col in fill_values:
                df[col] = df[col].fillna(fill_values[col])
            else:
                m = df[col].median()
                df[col] = df[col].fillna(m if pd.notna(m) else 0.0)

    df["publish_time"] = pd.to_datetime(df["publish_time"], utc=True, errors="coerce")
    df = _merge_cat_speed(df)

    cat_cols = ["cat_trend_p10", "cat_trend_p50", "cat_trend_p90", "cat_trend_spread"]
    for col in cat_cols:
        if fill_values and col in fill_values:
            df[col] = df[col].fillna(fill_values[col])
        else:
            df[col] = df[col].fillna(df[col].median())

    df["publish_hour"] = df["publish_time"].dt.hour.fillna(12).astype(int)
    df["publish_dayofweek"] = df["publish_time"].dt.dayofweek.fillna(0).astype(int)
    df["publish_period"] = df["publish_hour"].map(hour_bucket)
    df["is_weekend"] = (df["publish_dayofweek"] >= 5).astype(int)

    channel_features, title_features, tag_features, desc_features, cat_speed_features = _build_feature_groups(df)

    features = (
        ["category_id", "publish_hour", "publish_dayofweek", "publish_period", "is_weekend"]
        + title_features
        + tag_features
        + desc_features
        + channel_features
        + cat_speed_features
    )

    fill_out: Dict[str, float] = {}
    for col in channel_cols + cat_cols:
        if col in df.columns:
            fill_out[col] = float(df[col].median())

    return df, features, fill_out


def train_and_save(
    csv_path: Path | str = DEFAULT_TRAINING_CSV,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    meta_path: Path | str = DEFAULT_META_PATH,
) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    csv_path = Path(csv_path)
    model_path = Path(model_path)
    meta_path = Path(meta_path)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Load & clean ===")
    df_raw = pd.read_csv(csv_path)
    df, features, fill_values = prepare_frame(df_raw)
    target = "trendy"
    if target not in df.columns:
        raise ValueError(f"Expected '{target}' column in training CSV")

    X = df[features].astype(float)
    y = df[target].astype(int)

    print(f"Samples: {len(df)}, features: {len(features)}")
    print(f"Class balance (1): {y.mean():.4f}")

    neg, pos = np.bincount(y.to_numpy(dtype=int))
    scale_pos_weight = float(neg / pos)
    print(f"scale_pos_weight = {scale_pos_weight:.4f}")

    print("\n=== Hyperparameter tuning (RandomizedSearchCV) ===")
    param_dist = {
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5, 7, 10],
        "gamma": [0, 0.5, 1, 2],
        "reg_alpha": [0, 0.1, 0.5, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }

    base_model = xgb.XGBClassifier(
        n_estimators=500,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="aucpr",
        verbosity=0,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=50,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X, y)
    best_params = dict(search.best_params_)
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print(f"Best params: {best_params}")

    print("\n=== Final fit with early stopping ===")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    model = xgb.XGBClassifier(
        n_estimators=500,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        verbosity=0,
        **best_params,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    best_it = int(model.best_iteration) if model.best_iteration is not None else 500
    best_it = max(best_it, 1)
    print(f"best_iteration (early stopping): {best_it}")

    print("\n=== Refit on full data ===")
    final_model = xgb.XGBClassifier(
        n_estimators=best_it,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="aucpr",
        verbosity=0,
        **best_params,
    )
    final_model.fit(X, y, verbose=False)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(model_path))
    print(f"Saved model: {model_path}")

    imp = final_model.feature_importances_
    top_pairs = sorted(zip(features, imp), key=lambda x: -x[1])[:5]
    virality_top_features = [
        {
            "feature": name,
            "importance": float(score),
            "hint": FEATURE_HINTS.get(
                name,
                f"`{name}` is one of the strongest global drivers in the trending model.",
            ),
        }
        for name, score in top_pairs
    ]

    meta: Dict[str, Any] = {
        "features": features,
        "fill_values": {k: float(v) for k, v in fill_values.items()},
        "scale_pos_weight": scale_pos_weight,
        "best_params": best_params,
        "best_iteration": best_it,
        "virality_top_features": virality_top_features,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved metadata: {meta_path}")

    return final_model, meta


# --- Inference (UI / API) ---
_model_cache: Optional[xgb.XGBClassifier] = None
_meta_cache: Optional[Dict[str, Any]] = None
_cached_paths: Optional[Tuple[str, str]] = None


def _load_artifacts(model_path: Path, meta_path: Path) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    global _model_cache, _meta_cache, _cached_paths
    key = (str(model_path), str(meta_path))
    if _model_cache is not None and _cached_paths == key:
        return _model_cache, _meta_cache  # type: ignore[return-value]

    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing meta file: {meta_path}. Run training first.")
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model file: {model_path}. Run training first.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    clf = xgb.XGBClassifier()
    clf.load_model(str(model_path))
    _model_cache, _meta_cache, _cached_paths = clf, meta, key
    return clf, meta


def _risk_label(p: float) -> str:
    if p >= 0.66:
        return "HIGH"
    if p >= 0.33:
        return "MEDIUM"
    return "LOW"


def _normalize_record_for_models(record: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested `video` / `channel` objects from dashboard payloads."""
    row = dict(record)
    if "video" in row and isinstance(row["video"], dict):
        v = row["video"]
        row.setdefault("title", v.get("title", ""))
        row.setdefault("category_id", v.get("categoryId", v.get("category_id")))
        row.setdefault("publish_time", v.get("publishedAt", v.get("publish_time")))
        row.setdefault("views", v.get("viewCount", v.get("views")))
        row.setdefault("likes", v.get("likeCount", v.get("likes")))
        row.setdefault("comments", v.get("commentCount", v.get("comments")))
        row.setdefault("channel_title", v.get("channelTitle", row.get("channel_title")))
        row.setdefault("channelId", v.get("channelId"))
    if "channel" in row and isinstance(row["channel"], dict):
        ch = row["channel"]
        row.setdefault("channel_views", ch.get("viewCount", ch.get("channel_views")))
        row.setdefault("channel_subscribers", ch.get("subscriberCount", ch.get("channel_subscribers")))
        row.setdefault("channel_videos", ch.get("videoCount", ch.get("channel_videos")))
        row.setdefault("channel_title", ch.get("title", row.get("channel_title")))
    return row


def predict_stage1_trending(
    record: Dict[str, Any],
    *,
    model_path: Path | str | None = None,
    meta_path: Path | str | None = None,
) -> Dict[str, Any]:
    """
    Run Stage-1 XGBoost on one video-shaped record.

    Expected keys (snake_case or YouTube-style camelCase):
      - category_id / categoryId
      - publish_time / publishedAt (ISO8601)
      - title
      - tags (optional, pipe-separated like CSV) — default "[none]"
      - description (optional)
      - channel_views / viewCount (channel total views, not video)
      - channel_subscribers / subscriberCount
      - channel_videos / videoCount
      - channel_title / channelTitle

    Returns JSON-serializable dict for UI:
      probability (0..1), probability_percent, risk_label, predicted_trendy,
      predicted_class (1=trendy), features_used
    """
    model_path = Path(model_path or DEFAULT_MODEL_PATH)
    meta_path = Path(meta_path or DEFAULT_META_PATH)
    model, meta = _load_artifacts(model_path, meta_path)

    features: List[str] = meta["features"]
    fill_values: Dict[str, float] = meta.get("fill_values", {})

    row = _normalize_record_for_models(dict(record))
    if "category_id" not in row and "categoryId" in row:
        row["category_id"] = row["categoryId"]
    if "publish_time" not in row and "publishedAt" in row:
        row["publish_time"] = row["publishedAt"]
    if "channel_views" not in row and "channel" in row and isinstance(row["channel"], dict):
        ch = row["channel"]
        row["channel_views"] = ch.get("viewCount", ch.get("view_count"))
        row["channel_subscribers"] = ch.get("subscriberCount", ch.get("subscriber_count"))
        row["channel_videos"] = ch.get("videoCount", ch.get("video_count"))
        if "channel_title" not in row and "title" in ch:
            row["channel_title"] = ch["title"]
    if "channel_views" not in row and "channelViewCount" in row:
        row["channel_views"] = row["channelViewCount"]
    if "channel_subscribers" not in row and "subscriberCount" in row:
        row["channel_subscribers"] = row["subscriberCount"]
    if "channel_videos" not in row and "videoCount" in row:
        row["channel_videos"] = row["videoCount"]
    if "channel_title" not in row and "channelTitle" in row:
        row["channel_title"] = row.get("channelTitle") or "unknown"
    if "channel_title" not in row:
        row["channel_title"] = row.get("channel_title", "unknown")
    row.setdefault("title", "")
    row.setdefault("tags", "[none]")
    row.setdefault("description", "")

    df = pd.DataFrame([row])
    df, _, _ = prepare_frame(df, fill_values=fill_values)

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Engineered frame missing columns: {missing}")

    X = df[features].astype(float)
    proba = float(model.predict_proba(X)[0, 1])
    predicted_trendy = proba >= 0.5
    return {
        "probability": round(proba, 6),
        "probability_percent": int(round(proba * 100)),
        "risk_label": _risk_label(proba),
        "predicted_trendy": predicted_trendy,
        "predicted_class": int(predicted_trendy),
        "features_used": features,
    }


def _virality_hints_from_meta(model: xgb.XGBClassifier, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    cached = meta.get("virality_top_features")
    if isinstance(cached, list) and len(cached) >= 5:
        return list(cached)[:5]
    feats: List[str] = meta["features"]
    imp = model.feature_importances_
    top = sorted(zip(feats, imp), key=lambda x: -x[1])[:5]
    return [
        {
            "feature": name,
            "importance": float(score),
            "hint": FEATURE_HINTS.get(
                name,
                f"`{name}` is one of the strongest global drivers in the trending model.",
            ),
        }
        for name, score in top
    ]


def predict_trending_pipeline(
    record: Dict[str, Any],
    *,
    stage1_model_path: Path | str | None = None,
    stage1_meta_path: Path | str | None = None,
    stage2_model_path: Path | str | None = None,
    stage2_meta_path: Path | str | None = None,
) -> Dict[str, Any]:
    """
    Run Stage 1. If predicted trendy, run Stage 2 (post-trend trajectory).
    If not predicted trendy, attach top-5 global Stage-1 feature hints only.

    Returns:
      stage1: Stage-1 scores
      stage2: Stage-2 post-trend class (when Stage 1 predicts trending)
      virality_hints: top drivers + hints (when Stage 1 does not predict trending)
    """
    row = _normalize_record_for_models(dict(record))
    s1 = predict_stage1_trending(
        row,
        model_path=stage1_model_path or DEFAULT_MODEL_PATH,
        meta_path=stage1_meta_path or DEFAULT_META_PATH,
    )
    out: Dict[str, Any] = {
        "stage1": s1,
        "stage2": None,
        "virality_hints": None,
    }

    model_path = Path(stage1_model_path or DEFAULT_MODEL_PATH)
    meta_path = Path(stage1_meta_path or DEFAULT_META_PATH)
    model, meta = _load_artifacts(model_path, meta_path)

    if not s1["predicted_trendy"]:
        out["virality_hints"] = _virality_hints_from_meta(model, meta)
        return out

    from stage_2 import predict_stage2_post_trend

    out["stage2"] = predict_stage2_post_trend(
        row,
        model_path=stage2_model_path or DEFAULT_STAGE2_MODEL_PATH,
        meta_path=stage2_meta_path or DEFAULT_STAGE2_META_PATH,
    )
    return out


def main() -> None:
    train_and_save()


if __name__ == "__main__":
    main()
