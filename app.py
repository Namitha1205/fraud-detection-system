from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
data = pd.read_csv(url)

# Prepare data
X = data.drop('Class', axis=1)
y = data['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=2000, class_weight='balanced')
model.fit(X_scaled, y)

@app.route('/')
def home():
    return "Fraud Detection API is running"

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json['data']
    
    input_scaled = scaler.transform([input_data])
    prob = model.predict_proba(input_scaled)[0][1]

    if prob > 0.8:
        risk = "HIGH RISK"
    elif prob > 0.5:
        risk = "MEDIUM RISK"
    else:
        risk = "LOW RISK"

    return jsonify({
        "risk_score": float(prob),
        "status": risk
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)