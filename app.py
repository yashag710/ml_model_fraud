from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})  # Enable CORS for frontend

# Load the trained model
with open("fraud_detection_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Expecting JSON input
        print("Received data:", data)

        # Extracting input features
        input_data = {
            "payer_id": [data.get("payer_id")],
            "amount": [float(data.get("amount"))],  # Convert to float
            "ip_address": [data.get("ip_address")],
            "state": [data.get("state")],
            "failed_attempt": [int(data.get("failed_attempt"))]  # Convert to int
        }

        # Convert to DataFrame
        input_df = pd.DataFrame(input_data)
        print("Input DataFrame:", input_df)

        # Make prediction
        prediction = model.predict(input_df)[0]
        print("Prediction:", prediction)

        return jsonify({"fraudulent": bool(prediction)})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(port=5001, debug=True)