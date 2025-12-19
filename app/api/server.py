"""
Yatri Travel Assistant - Flask Web Server
"""

import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
sys.path.insert(0, str(Path(__file__).parent.parent / "config"))

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from rag_engine import create_rag_pipeline
import requests

# Template folder is in ../web/templates relative to this file
template_dir = Path(__file__).parent.parent / "web" / "templates"
static_dir = Path(__file__).parent.parent / "web" / "static"

app = Flask(__name__, template_folder=str(template_dir), static_folder=str(static_dir))
CORS(app)

# Initialize RAG pipeline
rag_pipeline = None

# Popular Indian destinations with coordinates
DESTINATIONS = {
    "manali": {"lat": 32.2432, "lon": 77.1892, "state": "Himachal Pradesh"},
    "shimla": {"lat": 31.1048, "lon": 77.1734, "state": "Himachal Pradesh"},
    "dharamshala": {"lat": 32.2190, "lon": 76.3234, "state": "Himachal Pradesh"},
    "kullu": {"lat": 31.9579, "lon": 77.1095, "state": "Himachal Pradesh"},
    "dalhousie": {"lat": 32.5387, "lon": 75.9707, "state": "Himachal Pradesh"},
    "leh": {"lat": 34.1526, "lon": 77.5771, "state": "Ladakh"},
    "ladakh": {"lat": 34.1526, "lon": 77.5771, "state": "Ladakh"},
    "srinagar": {"lat": 34.0837, "lon": 74.7973, "state": "Jammu & Kashmir"},
    "gulmarg": {"lat": 34.0484, "lon": 74.3805, "state": "Jammu & Kashmir"},
    "pahalgam": {"lat": 34.0161, "lon": 75.3150, "state": "Jammu & Kashmir"},
    "rishikesh": {"lat": 30.0869, "lon": 78.2676, "state": "Uttarakhand"},
    "haridwar": {"lat": 29.9457, "lon": 78.1642, "state": "Uttarakhand"},
    "mussoorie": {"lat": 30.4598, "lon": 78.0644, "state": "Uttarakhand"},
    "nainital": {"lat": 29.3919, "lon": 79.4542, "state": "Uttarakhand"},
    "dehradun": {"lat": 30.3165, "lon": 78.0322, "state": "Uttarakhand"},
    "kedarnath": {"lat": 30.7346, "lon": 79.0669, "state": "Uttarakhand"},
    "badrinath": {"lat": 30.7433, "lon": 79.4938, "state": "Uttarakhand"},
    "almora": {"lat": 29.5971, "lon": 79.6591, "state": "Uttarakhand"},
    "delhi": {"lat": 28.6139, "lon": 77.2090, "state": "Delhi"},
    "jaipur": {"lat": 26.9124, "lon": 75.7873, "state": "Rajasthan"},
    "udaipur": {"lat": 24.5854, "lon": 73.7125, "state": "Rajasthan"},
    "agra": {"lat": 27.1767, "lon": 78.0081, "state": "Uttar Pradesh"},
    "varanasi": {"lat": 25.3176, "lon": 82.9739, "state": "Uttar Pradesh"},
    "mumbai": {"lat": 19.0760, "lon": 72.8777, "state": "Maharashtra"},
    "goa": {"lat": 15.2993, "lon": 74.1240, "state": "Goa"},
    "kochi": {"lat": 9.9312, "lon": 76.2673, "state": "Kerala"},
    "munnar": {"lat": 10.0889, "lon": 77.0595, "state": "Kerala"},
}


def init_rag():
    global rag_pipeline
    print("Initializing Yatri RAG pipeline...")
    rag_pipeline = create_rag_pipeline()
    if rag_pipeline:
        print("RAG pipeline ready")
    else:
        print("Failed to initialize RAG pipeline")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    global rag_pipeline
    if not rag_pipeline:
        return jsonify({"error": "Service unavailable", "response": ""}), 503

    data = request.get_json()
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Empty message", "response": ""}), 400

    try:
        result = rag_pipeline.chat(message)
        response_text = result.get("response", "")
        return jsonify({"response": response_text, "error": None})
    except Exception as e:
        return jsonify({"error": str(e), "response": ""}), 500


@app.route("/api/weather", methods=["GET"])
def get_weather():
    """Get weather for a destination using Open-Meteo (free, no API key)"""
    city = request.args.get("city", "").lower().strip()
    
    if not city:
        return jsonify({"error": "City parameter required"}), 400
    
    # Find destination coordinates
    dest = DESTINATIONS.get(city)
    if not dest:
        # Try partial match
        for name, data in DESTINATIONS.items():
            if city in name or name in city:
                dest = data
                city = name
                break
    
    if not dest:
        return jsonify({"error": f"City '{city}' not found. Try: Manali, Shimla, Leh, etc."}), 404
    
    try:
        # Open-Meteo API (free, no key needed)
        url = f"https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": dest["lat"],
            "longitude": dest["lon"],
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,weather_code",
            "timezone": "Asia/Kolkata",
            "forecast_days": 5
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if "current" not in data:
            return jsonify({"error": "Weather data unavailable"}), 503
        
        # Weather code to description mapping
        weather_codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Depositing rime fog",
            51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
            95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail"
        }
        
        weather_icons = {
            0: "☀️", 1: "🌤️", 2: "⛅", 3: "☁️",
            45: "🌫️", 48: "🌫️",
            51: "🌦️", 53: "🌧️", 55: "🌧️",
            61: "🌧️", 63: "🌧️", 65: "🌧️",
            71: "🌨️", 73: "🌨️", 75: "❄️",
            80: "🌦️", 81: "🌧️", 82: "⛈️",
            95: "⛈️", 96: "⛈️", 99: "⛈️"
        }
        
        current = data["current"]
        daily = data["daily"]
        code = current["weather_code"]
        
        result = {
            "city": city.title(),
            "state": dest["state"],
            "current": {
                "temp": round(current["temperature_2m"]),
                "humidity": current["relative_humidity_2m"],
                "wind": round(current["wind_speed_10m"]),
                "condition": weather_codes.get(code, "Unknown"),
                "icon": weather_icons.get(code, "🌡️")
            },
            "forecast": []
        }
        
        # Add 5-day forecast
        for i in range(min(5, len(daily["time"]))):
            day_code = daily["weather_code"][i]
            result["forecast"].append({
                "date": daily["time"][i],
                "high": round(daily["temperature_2m_max"][i]),
                "low": round(daily["temperature_2m_min"][i]),
                "condition": weather_codes.get(day_code, "Unknown"),
                "icon": weather_icons.get(day_code, "🌡️")
            })
        
        return jsonify(result)
        
    except requests.Timeout:
        return jsonify({"error": "Weather service timeout"}), 503
    except Exception as e:
        return jsonify({"error": f"Weather fetch failed: {str(e)}"}), 500


@app.route("/api/destinations", methods=["GET"])
def list_destinations():
    """List available destinations for weather"""
    return jsonify({
        "destinations": [
            {"name": name.title(), "state": data["state"]} 
            for name, data in sorted(DESTINATIONS.items())
        ]
    })


@app.route("/api/health")
def health():
    global rag_pipeline
    if not rag_pipeline:
        return jsonify({"status": "error", "message": "RAG not initialized"})

    health = rag_pipeline.get_health_status()
    return jsonify({
        "status": "ok" if health.get("initialized") else "error",
        "vector_store": health.get("vector_store", False),
        "embedding_model": health.get("embedding_model", False),
        "ollama": health.get("ollama", False),
        "total_embeddings": health.get("total_embeddings", 0),
    })


def main():
    init_rag()
    print("\n" + "=" * 50)
    print("Yatri Travel Assistant")
    print("=" * 50)
    print("Open http://localhost:5000 in your browser")
    print("Press Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
