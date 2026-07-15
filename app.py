# ==========================================================
# RISING WATERS - FLOOD PREDICTION SYSTEM
# app.py
# ==========================================================

from flask import Flask, render_template, request
import pandas as pd
import joblib

# ==========================================================
# LOAD TRAINED FILES
# ==========================================================

model = joblib.load("models/flood_prediction_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")
model_info = joblib.load("models/model_info.pkl")

# ==========================================================
# CREATE FLASK APP
# ==========================================================

app = Flask(__name__)

# ==========================================================
# HOME PAGE
# ==========================================================

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/form")
def form():
    return render_template("index.html")

# ==========================================================
# PREDICTION
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    # -------------------------
    # Get User Inputs
    # -------------------------

    temp = float(request.form["Temp"])
    humidity = float(request.form["Humidity"])
    cloud_cover = float(request.form["Cloud_Cover"])
    annual = float(request.form["ANNUAL"])
    jan_feb = float(request.form["Jan_Feb"])
    mar_may = float(request.form["Mar_May"])
    jun_sep = float(request.form["Jun_Sep"])
    oct_dec = float(request.form["Oct_Dec"])
    avgjune = float(request.form["avgjune"])
    sub = float(request.form["sub"])

    # -------------------------
    # Create Input DataFrame
    # -------------------------

    input_data = pd.DataFrame([{
        "Temp": temp,
        "Humidity": humidity,
        "Cloud Cover": cloud_cover,
        "ANNUAL": annual,
        "Jan-Feb": jan_feb,
        "Mar-May": mar_may,
        "Jun-Sep": jun_sep,
        "Oct-Dec": oct_dec,
        "avgjune": avgjune,
        "sub": sub
    }])

    # Keep feature order exactly the same as training
    input_data = input_data[feature_columns]

    # Scale input
    input_scaled = scaler.transform(input_data)
        # ==========================================================
    # MAKE PREDICTION
    # ==========================================================

    prediction = model.predict(input_scaled)[0]

    probabilities = model.predict_proba(input_scaled)[0]
    print("Prediction:", prediction)
    print("Probabilities:", probabilities)

    no_flood_probability = probabilities[0] * 100
    flood_probability = probabilities[1] * 100

    # ==========================================================
    # RESULT
    # ==========================================================

    if prediction == 1:

        result = "FLOOD LIKELY"

        risk_level = "HIGH"

        reasons = [
            "Heavy rainfall detected.",
            "High humidity level.",
            "Cloud cover is high.",
            "Weather conditions indicate possible flooding."
        ]

        suggestions = [
            "Avoid low-lying areas.",
            "Follow weather alerts.",
            "Prepare emergency supplies.",
            "Move to safer locations if necessary."
        ]

    else:

        result = "NO FLOOD EXPECTED"

        risk_level = "LOW"

        reasons = [
            "Weather conditions are stable.",
            "Rainfall is within normal limits.",
            "Flood risk is currently low."
        ]

        suggestions = [
            "Continue monitoring weather updates.",
            "Maintain drainage systems.",
            "Stay prepared during rainy seasons."
        ]

    # ==========================================================
    # MODEL INFORMATION
    # ==========================================================

    best_model = model_info["best_model"]
    accuracy = model_info["accuracy"] * 100

    # ==========================================================
    # RETURN RESULT
    # ==========================================================

    return render_template(
        "result.html",

        prediction=result,

        flood_probability=round(flood_probability, 2),

        no_flood_probability=round(no_flood_probability, 2),

        risk_level=risk_level,

        reasons=reasons,

        suggestions=suggestions,

        best_model=best_model,

        accuracy=round(accuracy, 2)
    )

# ==========================================================
# RUN FLASK
# ==========================================================

if __name__ == "__main__":
    app.run(debug=True)