"""
Intelligent Credit Risk Scoring System — Flask REST API
========================================================
Endpoints:
  GET  /           → Web UI (index.html)
  POST /predict    → JSON prediction endpoint
  GET  /health     → Health check
"""

import json
import os
import gc
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import xgboost as xgb
from pathlib import Path
from scipy.stats import rankdata
from flask import Flask, request, jsonify, render_template

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
# Repo root is /opt/render/project/src/ when using flask_app.app:app
BASE_DIR     = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", str(BASE_DIR / "model_artifacts")))

# On free Render (512 MB RAM), load 3 folds instead of 5 to stay within memory
MAX_FOLDS = int(os.environ.get("MAX_FOLDS", "3"))

app = Flask(__name__)


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────
def prob_to_credit_score(probability, min_score=300, max_score=850):
    """Convert default probability → FICO-style score (300–850)."""
    return np.round(
        min_score + (1 - probability) * (max_score - min_score)
    ).astype(int)


def assign_risk_band(score):
    """Return (band_label, color, recommendation, interest_rate) for a score."""
    if score >= 750:
        return "Excellent", "#27ae60", "APPROVE — Prime Rate", "8–10%"
    elif score >= 700:
        return "Good", "#2ecc71", "APPROVE — Standard Rate", "11–14%"
    elif score >= 650:
        return "Fair", "#f39c12", "APPROVE — Higher Rate", "15–19%"
    elif score >= 600:
        return "Poor", "#e67e22", "CONDITIONAL — Risk Premium", "20–25%"
    elif score >= 550:
        return "Very Poor", "#e74c3c", "DECLINE / Require Collateral", "N/A"
    else:
        return "High Risk", "#8e44ad", "DECLINE", "N/A"


# ─────────────────────────────────────────────
# CreditRiskScorer
# ─────────────────────────────────────────────
class CreditRiskScorer:
    """
    Production credit risk scoring engine.
    Loads all artifacts once at startup; thread-safe for inference.
    """

    def __init__(self, lgb_models, xgb_models, meta_learner,
                 imputer, feature_list, ensemble_weights, metadata):
        self.lgb_models       = lgb_models
        self.xgb_models       = xgb_models
        self.meta_learner     = meta_learner
        self.imputer          = imputer
        self.feature_list     = feature_list
        self.ensemble_weights = ensemble_weights
        self.metadata         = metadata
        self.threshold        = metadata.get("optimal_threshold", 0.15)

    @classmethod
    def load(cls, artifact_dir, max_folds=3):
        artifact_dir = Path(artifact_dir)

        print(f"[CreditRiskScorer] Loading from : {artifact_dir.resolve()}")
        print(f"[CreditRiskScorer] Dir exists   : {artifact_dir.exists()}")
        if artifact_dir.exists():
            print(f"[CreditRiskScorer] Files found  : {[f.name for f in sorted(artifact_dir.iterdir())]}")

        with open(artifact_dir / "feature_list.json")     as f: feature_list     = json.load(f)
        with open(artifact_dir / "model_metadata.json")   as f: metadata         = json.load(f)
        with open(artifact_dir / "ensemble_weights.json") as f: ensemble_weights  = json.load(f)

        imputer      = joblib.load(artifact_dir / "imputer.pkl")
        meta_learner = joblib.load(artifact_dir / "meta_learner.pkl")

        # Load only max_folds to stay within free-tier RAM (512 MB)
        n_folds = min(max_folds, metadata["n_folds"])

        lgb_models = [
            lgb.Booster(model_file=str(artifact_dir / f"lgbm_fold_{i+1}.txt"))
            for i in range(n_folds)
        ]
        xgb_models = [
            joblib.load(artifact_dir / f"xgb_fold_{i+1}.pkl")
            for i in range(n_folds)
        ]

        # Free memory immediately after loading
        gc.collect()

        print(f"[CreditRiskScorer] Loaded — {n_folds}x LGB + {n_folds}x XGB + meta-learner")
        return cls(lgb_models, xgb_models, meta_learner,
                   imputer, feature_list, ensemble_weights, metadata)

    def predict_proba(self, X):
        X_aligned = X.reindex(columns=self.feature_list, fill_value=np.nan)
        X_imp     = pd.DataFrame(
            self.imputer.transform(X_aligned), columns=self.feature_list
        )
        lgb_preds   = np.mean([m.predict(X_imp) for m in self.lgb_models], axis=0)
        xgb_preds   = np.mean([m.predict(xgb.DMatrix(X_imp)) for m in self.xgb_models], axis=0)
        stack_preds = self.meta_learner.predict_proba(
            np.column_stack([lgb_preds, xgb_preds])
        )[:, 1]

        w = self.ensemble_weights
        weights = [w.get("lgb_tuned", 0.4), w.get("xgboost", 0.3), w.get("stacking", 0.3)]
        ranked  = [rankdata(p) / len(p) for p in [lgb_preds, xgb_preds, stack_preds]]
        return sum(wt * r for wt, r in zip(weights, ranked))

    def score_report(self, X):
        proba  = self.predict_proba(X)
        scores = prob_to_credit_score(proba)
        results = []
        for i, (p, s) in enumerate(zip(proba, scores)):
            band, color, rec, rate = assign_risk_band(int(s))
            results.append({
                "applicant_index"    : i,
                "default_probability": round(float(p), 4),
                "credit_score"       : int(s),
                "risk_band"          : band,
                "band_color"         : color,
                "decision"           : "REJECT" if p >= self.threshold else "APPROVE",
                "interest_rate"      : rate,
                "recommendation"     : rec,
                "high_risk_flag"     : int(p >= self.threshold),
            })
        return results


# ─────────────────────────────────────────────
# Load scorer at MODULE LEVEL
# Runs when Gunicorn imports this module — NOT inside __main__
# ─────────────────────────────────────────────
scorer = None

print(f"[app] ARTIFACT_DIR = {ARTIFACT_DIR.resolve()}")
print(f"[app] MAX_FOLDS    = {MAX_FOLDS}")

try:
    scorer = CreditRiskScorer.load(ARTIFACT_DIR, max_folds=MAX_FOLDS)
    print("[app] CreditRiskScorer ready.")
except Exception as e:
    print(f"[app] WARNING: Could not load scorer — {e}")
    scorer = None


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status" : "ok",
        "model"  : "loaded" if scorer else "not_loaded",
        "version": scorer.metadata.get("model_type", "N/A") if scorer else "N/A",
        "folds"  : len(scorer.lgb_models) if scorer else 0,
    })


@app.route("/predict", methods=["POST"])
def predict():
    if scorer is None:
        return jsonify({"error": "Model not loaded. Check model_artifacts/ directory."}), 503

    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Empty or invalid JSON body."}), 400

        records = data if isinstance(data, list) else [data]
        df      = pd.DataFrame(records)

        report = scorer.score_report(df)
        return jsonify({
            "success"     : True,
            "predictions" : report,
            "n_applicants": len(report),
        })

    except Exception as e:
        print(f"[predict] ERROR: {e}")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# Entry Point (local only — Gunicorn ignores this)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
