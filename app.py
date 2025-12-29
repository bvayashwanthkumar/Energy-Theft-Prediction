from flask import Flask, request, jsonify
import joblib
import pandas as pd

from models.features import FEATURES
from flask import render_template

app = Flask(__name__)

@app.route('/ui')
def ui():
    return render_template('index.html')

theft_model = joblib.load('models/theft_model.pkl')
anomaly_model = joblib.load('models/anomaly_model.pkl')

print("Models loaded successfully")

@app.route('/predict-theft', methods=['POST'])
def predict_theft():
    input_data = request.json
    df = pd.DataFrame([input_data])

    # Ensure column order and missing safety
    df = df.reindex(columns=FEATURES, fill_value=0)

    # 🔴 RULE-BASED HIGH-RISK OVERRIDE (FOR DEMO + REALISM)
    if (
        df['Global_active_power'].iloc[0] > 5.0 and
        df['Global_intensity'].iloc[0] > 20 and
        df['Voltage'].iloc[0] < 210
    ):
        return jsonify({
            'theft_prediction': 1
        })

    # ML prediction (fallback)
    prediction = theft_model.predict(df)[0]

    return jsonify({
        'theft_prediction': int(prediction)
    })


@app.route('/detect-anomaly', methods=['POST'])
def detect_anomaly():
    input_data = request.json
    df = pd.DataFrame([input_data])
    anomaly = anomaly_model.predict(df[FEATURES])[0]
    anomaly = 1 if anomaly == -1 else 0
    return jsonify({
        'anomaly_detected': anomaly
    })

@app.route('/forecast-demand', methods=['GET'])
def forecast_demand():
    forecast = pd.read_csv('data/processed/demand_forecast.csv')
    return forecast.to_json()

if __name__ == '__main__':
    app.run(debug=True)
