# app.py
import pandas as pd
import gradio as gr
import pickle
from sklearn.preprocessing import StandardScaler

# Modelle laden
with open("grammy_rf_model.pkl", "rb") as f_rf:
    rf_bundle = pickle.load(f_rf)

with open("grammy_lr_model.pkl", "rb") as f_lr:
    lr_bundle = pickle.load(f_lr)

# Modelle und Feature-Namen
models = {
    "Random Forest": rf_bundle,
    "Logistic Regression": lr_bundle
}

# Vorhersagefunktion
def predict_grammy(model_choice, danceability, energy, valence, bpm):
    bundle = models[model_choice]
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    scaler = bundle["scaler"]

    input_data = pd.DataFrame([[danceability, energy, valence, bpm]], columns=feature_names)
    input_scaled = scaler.transform(input_data)
    proba = model.predict_proba(input_scaled)[0][1]  # Wahrscheinlichkeit für "winner = True"
    return f"Gewinnwahrscheinlichkeit: {proba * 100:.2f}%"

# Gradio UI definieren
demo = gr.Interface(
    fn=predict_grammy,
    inputs=[
        gr.Dropdown(choices=["Random Forest", "Logistic Regression"], label="Modell auswählen"),
        gr.Slider(0, 100, step=1, label="Danceability (0-100)"),
        gr.Slider(0, 100, step=1, label="Energy (0-100)"),
        gr.Slider(0, 100, step=1, label="Valence (Stimmung) (0-100)"),
        gr.Number(label="Tempo (Beats Per Minute)")
    ],
    outputs="text",
    title="Grammy Gewinnvorhersage",
    description="Wähle ein Modell und gib musikalische Eigenschaften eines Songs ein, um die Grammy-Gewinnwahrscheinlichkeit zu berechnen."
)

# Starten
demo.launch(share=True)
