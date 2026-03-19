from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os

app = Flask(__name__)

# ------------------ LOAD DATA ------------------

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
data = pd.read_csv(url)

# ------------------ PREPROCESS ------------------

X = data.drop('Class', axis=1)
y = data['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------ TRAIN MODEL ------------------

model = LogisticRegression(max_iter=2000, class_weight='balanced')
model.fit(X_scaled, y)

# ------------------ ACCURACY ------------------

y_pred = model.predict(X_scaled)
accuracy = accuracy_score(y, y_pred)

# ------------------ ROUTES ------------------

@app.route('/')
def home():
    return f"""
    <html>
    <head>
        <title>Fraud Detection System</title>
    </head>
    <body style="font-family: Arial; text-align:center; margin-top:50px;">

        <h1>Fraud Detection System</h1>

        <p><b>Model:</b> Logistic Regression</p>
        <p><b>Accuracy:</b> {round(accuracy * 100, 2)}%</p>

        <p>Enter 30 transaction values (comma separated):</p>

        <input type="text" id="inputData" size="80"
        value="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0">

        <br><br>

        <button onclick="predict()">Predict</button>

        <h3 id="result"></h3>

        <script>
            async function predict() {{
                let input = document.getElementById("inputData").value.split(",").map(Number);

                let response = await fetch("/predict", {{
                    method: "POST",
                    headers: {{
                        "Content-Type": "application/json"
                    }},
                    body: JSON.stringify({{ data: input }})
                }});

                let result = await response.json();

                document.getElementById("result").innerText =
                    "Risk Score: " + result.risk_score +
                    " | Status: " + result.status;
            }}
        </script>

    </body>
    </html>
    """

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

# ------------------ RUN APP ------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)