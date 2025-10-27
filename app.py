from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
import os
import requests

app = Flask(__name__)
CORS(app)



def read_image_from_filestorage(fs):
    try:
        img = Image.open(fs).convert("RGB")
        return np.array(img)
    except Exception:
        return None

def normalize_ndvi(img):
    
    return img.astype(np.float32) / 255.0

def estimate_health_from_ndvi(ndvi):
    mean_ndvi = np.mean(ndvi)
    overall_health = "Healthy" if mean_ndvi > 0.5 else "Poor"
    pct_poor = float(np.sum(ndvi < 0.3)) / ndvi.size
    return {
        "overall_health": overall_health,
        "mean_ndvi": float(mean_ndvi),
        "pct_poor": float(pct_poor)
    }

def detect_hotspots(ndvi):
    
    return [
        {"risk_score": 0.8, "mean_ndvi": 0.2, "area": 120},
        {"risk_score": 0.6, "mean_ndvi": 0.25, "area": 80}
    ]

def create_heatmap_png(ndvi):
    ndvi_min, ndvi_max = ndvi.min(), ndvi.max()
    if ndvi_max == ndvi_min:
        ndvi_norm = np.zeros_like(ndvi, dtype=np.uint8)
    else:
        ndvi_norm = ((ndvi - ndvi_min) / (ndvi_max - ndvi_min) * 255).astype(np.uint8)

    heatmap_img = cv2.applyColorMap(ndvi_norm, cv2.COLORMAP_JET)
    pil_img = Image.fromarray(heatmap_img)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf



@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    fs = request.files["image"]
    img = read_image_from_filestorage(fs)
    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    ndvi = normalize_ndvi(img)
    health = estimate_health_from_ndvi(ndvi)
    hotspots = detect_hotspots(ndvi)
    heatmap_buf = create_heatmap_png(ndvi)
    heatmap_b64 = base64.b64encode(heatmap_buf.getvalue()).decode("ascii")

    return jsonify({
        "health": health,
        "hotspots": hotspots,
        "heatmap_base64": heatmap_b64
    })

@app.route("/weather", methods=["GET"])
def get_weather():
    lat = request.args.get("lat")
    lon = request.args.get("lon")
    api_key = os.getenv("OPENWEATHER_API_KEY")


    if not lat or not lon:
        return jsonify({"error": "lat and lon query params required"}), 400
    if not api_key:
        return jsonify({"error": "OpenWeatherMap API key not configured"}), 400

    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": "Weather API failed", "details": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
