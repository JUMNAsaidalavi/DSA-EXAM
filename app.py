from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("eurovision_model.pkl")
model_columns = joblib.load("model_columns.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_data = {
        "Year": 2024,
        "Artist.gender": "Male",
        "Group.Solo": request.form["Group.Solo"],
        "Song.In.English": request.form["Song.In.English"],
        "danceability": float(request.form["danceability"]),
        "energy": float(request.form["energy"]),
        "tempo": 120,
        "acousticness": 0.3,
        "liveness": 0.2,
        "speechiness": 0.05
    }

    df_input = pd.DataFrame([input_data])
    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(df_input)[0]

    return f"Predicted Eurovision Points: {round(prediction,2)}"

if __name__ == "__main__":
    app.run(debug=True)