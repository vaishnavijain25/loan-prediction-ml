from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# ✅ Load all artifacts
model = pickle.load(open("artifacts/model.pkl", "rb"))
scaler = pickle.load(open("artifacts/scaler.pkl", "rb"))
encoders = pickle.load(open("artifacts/encoders.pkl", "rb"))
feature_columns = pickle.load(open("artifacts/feature_columns.pkl", "rb"))

@app.route("/")
def home():
    return "Loan Prediction API is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        input_data = []

        # ✅ FIXED ORDER
        for col in feature_columns:
            value = data.get(col)

            # ✅ Apply encoding if needed
            if col in encoders:
                le = encoders[col]
                value = le.transform([str(value)])[0]

            input_data.append(value)

        # Convert to numpy
        input_array = np.array(input_data).reshape(1, -1)

        # Scale
        input_scaled = scaler.transform(input_array)

        # Predict
        prediction = model.predict(input_scaled)

        return jsonify({
            "prediction": int(prediction[0])
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

if __name__ == "__main__":
    app.run(debug=True)
