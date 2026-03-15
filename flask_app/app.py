"""
Intelligent Credit Risk Scoring System — Flask REST API
Local development version — 5-fold LightGBM ensemble.
Threshold set to 0.50 for single-row demo predictions.
"""

import json
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from flask import Flask, request, jsonify, render_template

BASE_DIR     = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", str(BASE_DIR / "model_artifacts")))

# ── Demo threshold — 0.50 works correctly for single-row predictions ──────────
# (0.15 was optimized on full 307K training set, not single rows)
DEMO_THRESHOLD = 0.50

app = Flask(__name__)


def prob_to_credit_score(probability, min_score=300, max_score=850):
    return np.round(min_score + (1 - probability) * (max_score - min_score)).astype(int)


def assign_risk_band(score):
    if score >= 750: return "Excellent", "#10b981", "APPROVE — Prime Rate",         "8–10%"
    if score >= 700: return "Good",      "#34d399", "APPROVE — Standard Rate",      "11–14%"
    if score >= 650: return "Fair",      "#f59e0b", "APPROVE — Higher Rate",        "15–19%"
    if score >= 600: return "Poor",      "#f97316", "CONDITIONAL — Risk Premium",   "20–25%"
    if score >= 550: return "Very Poor", "#ef4444", "DECLINE / Require Collateral", "N/A"
    return             "High Risk",      "#8b5cf6", "DECLINE",                      "N/A"


# ── Load 5 tuned LightGBM folds ───────────────────────────────────────────────
scorer = None

print(f"[app] ARTIFACT_DIR = {ARTIFACT_DIR.resolve()}")

try:
    with open(ARTIFACT_DIR / "feature_list.json")   as f: feature_list = json.load(f)
    with open(ARTIFACT_DIR / "model_metadata.json") as f: metadata     = json.load(f)

    n_folds = metadata["n_folds"]

    lgb_models = [
        lgb.Booster(model_file=str(ARTIFACT_DIR / f"lgbm_fold_{i+1}.txt"))
        for i in range(n_folds)
    ]

    scorer = {
        "models"   : lgb_models,
        "features" : feature_list,
        "threshold": DEMO_THRESHOLD,
        "metadata" : metadata,
    }

    print(f"[app] Loaded — {n_folds}x LightGBM folds")
    print(f"[app] Threshold = {DEMO_THRESHOLD} | Features = {len(feature_list)}")

except Exception as e:
    print(f"[app] WARNING: {e}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model" : "loaded" if scorer else "not_loaded",
        "folds" : len(scorer["models"]) if scorer else 0,
    })


@app.route("/predict", methods=["POST"])
def predict():
    if not scorer:
        return jsonify({"error": "Model not loaded."}), 503

    try:
        data    = request.get_json(force=True)
        records = data if isinstance(data, list) else [data]
        df      = pd.DataFrame(records)

        X     = df.reindex(columns=scorer["features"], fill_value=np.nan)
        proba = np.mean([m.predict(X) for m in scorer["models"]], axis=0)
        thr   = scorer["threshold"]
        scores  = prob_to_credit_score(proba)
        results = []

        for i, (p, s) in enumerate(zip(proba, scores)):
            band, color, rec, rate = assign_risk_band(int(s))
            results.append({
                "applicant_index"    : i,
                "default_probability": round(float(p), 4),
                "credit_score"       : int(s),
                "risk_band"          : band,
                "band_color"         : color,
                "decision"           : "REJECT" if p >= thr else "APPROVE",
                "interest_rate"      : rate,
                "recommendation"     : rec,
                "high_risk_flag"     : int(p >= thr),
            })

        return jsonify({"success": True, "predictions": results, "n_applicants": len(results)})

    except Exception as e:
        print(f"[predict] ERROR: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
