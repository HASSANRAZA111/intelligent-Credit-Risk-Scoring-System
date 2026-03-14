"""
Intelligent Credit Risk Scoring System — Flask REST API
Lightweight demo mode for free-tier deployment (512MB RAM)
"""

import json
import os
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from pathlib import Path
from flask import Flask, request, jsonify, render_template

BASE_DIR     = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", str(BASE_DIR / "model_artifacts")))

app = Flask(__name__)


def prob_to_credit_score(p, lo=300, hi=850):
    return int(round(lo + (1 - p) * (hi - lo)))


def assign_risk_band(score):
    if score >= 750: return "Excellent", "#27ae60", "APPROVE — Prime Rate",        "8–10%"
    if score >= 700: return "Good",      "#2ecc71", "APPROVE — Standard Rate",     "11–14%"
    if score >= 650: return "Fair",      "#f39c12", "APPROVE — Higher Rate",       "15–19%"
    if score >= 600: return "Poor",      "#e67e22", "CONDITIONAL — Risk Premium",  "20–25%"
    if score >= 550: return "Very Poor", "#e74c3c", "DECLINE / Require Collateral","N/A"
    return             "High Risk",      "#8e44ad", "DECLINE",                     "N/A"


# ── Load ONE LightGBM fold + imputer only ──────────────────────────────────────
scorer = None

try:
    print(f"[app] Loading from {ARTIFACT_DIR.resolve()}")

    with open(ARTIFACT_DIR / "feature_list.json")   as f: feature_list = json.load(f)
    with open(ARTIFACT_DIR / "model_metadata.json") as f: metadata     = json.load(f)

    imputer   = joblib.load(ARTIFACT_DIR / "imputer.pkl")
    lgb_model = lgb.Booster(model_file=str(ARTIFACT_DIR / "lgbm_fold_1.txt"))
    threshold = metadata.get("optimal_threshold", 0.15)

    scorer = {"model": lgb_model, "imputer": imputer,
              "features": feature_list, "threshold": threshold}

    print("[app] Ready — 1x LightGBM fold loaded.")

except Exception as e:
    print(f"[app] WARNING: {e}")


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model" : "loaded" if scorer else "not_loaded",
    })


@app.route("/predict", methods=["POST"])
def predict():
    if not scorer:
        return jsonify({"error": "Model not loaded."}), 503

    try:
        data    = request.get_json(force=True)
        records = data if isinstance(data, list) else [data]
        df      = pd.DataFrame(records)

        X = df.reindex(columns=scorer["features"], fill_value=np.nan)
        X = pd.DataFrame(scorer["imputer"].transform(X), columns=scorer["features"])

        proba = scorer["model"].predict(X)
        thr   = scorer["threshold"]

        results = []
        for p in proba:
            score                   = prob_to_credit_score(p)
            band, color, rec, rate  = assign_risk_band(score)
            results.append({
                "default_probability": round(float(p), 4),
                "credit_score"       : score,
                "risk_band"          : band,
                "band_color"         : color,
                "decision"           : "REJECT" if p >= thr else "APPROVE",
                "interest_rate"      : rate,
                "recommendation"     : rec,
                "high_risk_flag"     : int(p >= thr),
            })

        return jsonify({"success": True, "predictions": results})

    except Exception as e:
        print(f"[predict] ERROR: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
