"""
Intelligent Credit Risk Scoring System — Flask REST API
Ultra-lightweight demo mode — no imputer, LightGBM handles NaN natively
"""

import json
import os
import numpy as np
import lightgbm as lgb
from pathlib import Path
from flask import Flask, request, jsonify, render_template

BASE_DIR     = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", str(BASE_DIR / "model_artifacts")))

app = Flask(__name__)


def prob_to_credit_score(p, lo=300, hi=850):
    return int(round(lo + (1 - float(p)) * (hi - lo)))


def assign_risk_band(score):
    if score >= 750: return "Excellent", "#27ae60", "APPROVE — Prime Rate",         "8–10%"
    if score >= 700: return "Good",      "#2ecc71", "APPROVE — Standard Rate",      "11–14%"
    if score >= 650: return "Fair",      "#f39c12", "APPROVE — Higher Rate",        "15–19%"
    if score >= 600: return "Poor",      "#e67e22", "CONDITIONAL — Risk Premium",   "20–25%"
    if score >= 550: return "Very Poor", "#e74c3c", "DECLINE / Require Collateral", "N/A"
    return             "High Risk",      "#8e44ad", "DECLINE",                      "N/A"


# ── Load model — NO imputer, NO pandas, NO sklearn ────────────────────────────
model     = None
features  = None
threshold = 0.15

try:
    print(f"[app] Loading from {ARTIFACT_DIR.resolve()}")

    with open(ARTIFACT_DIR / "feature_list.json")   as f: features  = json.load(f)
    with open(ARTIFACT_DIR / "model_metadata.json") as f: meta      = json.load(f)

    threshold = meta.get("optimal_threshold", 0.15)
    model     = lgb.Booster(model_file=str(ARTIFACT_DIR / "lgbm_fold_1.txt"))

    print(f"[app] Ready — LightGBM loaded, {len(features)} features, threshold={threshold}")

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
        "model" : "loaded" if model else "not_loaded",
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 503

    try:
        data    = request.get_json(force=True)
        records = data if isinstance(data, list) else [data]

        results = []
        for record in records:
            # Build numpy row — LightGBM handles NaN natively, no imputer needed
            row = np.array(
                [float(record.get(f, np.nan)) for f in features],
                dtype=np.float32
            ).reshape(1, -1)

            p     = float(model.predict(row)[0])
            score = prob_to_credit_score(p)
            band, color, rec, rate = assign_risk_band(score)

            results.append({
                "default_probability": round(p, 4),
                "credit_score"       : score,
                "risk_band"          : band,
                "band_color"         : color,
                "decision"           : "REJECT" if p >= threshold else "APPROVE",
                "interest_rate"      : rate,
                "recommendation"     : rec,
                "high_risk_flag"     : int(p >= threshold),
            })

        return jsonify({"success": True, "predictions": results})

    except Exception as e:
        print(f"[predict] ERROR: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
