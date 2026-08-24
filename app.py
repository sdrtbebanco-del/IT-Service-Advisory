"""
IT Helpdesk Advisory System - Flask backend
"""
import os
import re
import logging
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_model = None
_vectorizer = None
_classes = None

ACTIONS = {
    "network": "Restart router and check cables. Verify WiFi/Ethernet connection.",
    "hardware": "Inspect hardware components. Check physical connections.",
    "software": "Reinstall or update the application. Run as administrator if needed.",
    "account": "Reset or verify account credentials. Contact admin for unlock.",
    "printer": "Check printer connection and ink levels. Reinstall driver if needed.",
}


def preprocess_text(text):
    if not text or not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip() or ""


def load_model():
    global _model, _vectorizer, _classes
    if _model is not None:
        return
    try:
        import joblib
        _model = joblib.load(os.path.join(MODEL_DIR, "helpdesk_model.pkl"))
        _vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
        _classes = joblib.load(os.path.join(MODEL_DIR, "classes.pkl"))
        logger.info("Model loaded successfully")
    except FileNotFoundError as e:
        logger.error("Model files not found. Run train_model.py first. %s", e)
        raise
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        raise


def predict(text):
    load_model()
    cleaned = preprocess_text(text)
    if not cleaned:
        return None, None, None
    try:
        X = _vectorizer.transform([cleaned])
        pred = _model.predict(X)[0]
        proba = _model.predict_proba(X)[0]
        idx = list(_classes).index(pred)
        confidence = float(proba[idx])
        return pred, ACTIONS.get(pred, "Please contact IT support."), confidence
    except Exception as e:
        logger.error("Prediction error: %s", e)
        return None, None, None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_page():
    prediction, action, confidence = None, None, None
    error = None
    try:
        issue = request.form.get("issue", "").strip()
        if not issue:
            error = "Please enter a description of your issue."
        else:
            prediction, action, confidence = predict(issue)
            if prediction is None and not error:
                error = "Unable to process your request. Try rephrasing."
    except Exception as e:
        logger.exception("Predict route error")
        error = "An error occurred. Please try again."
    return render_template(
        "index.html",
        prediction=prediction,
        action=action,
        confidence=confidence,
        error=error,
        issue=request.form.get("issue", ""),
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(force=False, silent=True) or {}
        text = (data.get("text") or data.get("issue") or request.form.get("text") or request.form.get("issue") or "").strip()
        if not text:
            return jsonify({"ok": False, "error": "Missing 'text' or 'issue'"}), 400
        prediction, action, confidence = predict(text)
        if prediction is None:
            return jsonify({"ok": False, "error": "Prediction failed"}), 500
        return jsonify({
            "ok": True,
            "category": prediction,
            "action": action,
            "confidence": round(confidence, 4),
            "confidence_percent": round(confidence * 100, 2),
        })
    except Exception as e:
        logger.exception("API predict error")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({"ok": False, "error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"ok": False, "error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8000)
