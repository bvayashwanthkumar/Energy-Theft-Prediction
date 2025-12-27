from flask import Flask, request, jsonify
import joblib
import pandas as pd

from models.features import FEATURES

app = Flask(__name__)

@app.route('/')
def home():
    return "Energy Theft Prediction API is running"

theft_model = joblib.load('models/theft_model.pkl')
anomaly_model = joblib.load('models/anomaly_model.pkl')

print("Models loaded successfully")

@app.route('/predict-theft', methods=['POST'])
def predict_theft():
    input_data = request.json
    df = pd.DataFrame([input_data])
    prediction = theft_model.predict(df[FEATURES])[0]
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
