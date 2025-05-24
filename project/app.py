import pandas as pd
import gradio as gr
import pickle
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

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

# Daten für Song-Vergleich laden
try:
    grammy_df = pd.read_csv("data/grammy_winners.csv")
    spotify_df = pd.read_csv("data/Spotify-2000.csv")
    
    # Namen vereinheitlichen
    grammy_df['song_or_album'] = grammy_df['song_or_album'].str.lower().str.strip()
    grammy_df['artist'] = grammy_df['artist'].str.lower().str.strip()
    spotify_df['Title'] = spotify_df['Title'].str.lower().str.strip()
    spotify_df['Artist'] = spotify_df['Artist'].str.lower().str.strip()
    
    # Spotify-Features für Vergleich vorbereiten
    spotify_features_df = spotify_df[['Title', 'Artist', 'Danceability', 'Energy', 'Valence', 'Beats Per Minute (BPM)']].dropna()
    
    DATA_AVAILABLE = True
    print("✅ CSV-Dateien erfolgreich geladen")
    
except Exception as e:
    print(f"❌ CSV-Dateien nicht gefunden: {e}")
    DATA_AVAILABLE = False

# Vorhersagefunktion mit ähnlichen Songs und Grammy-Status
def predict_grammy(model_choice, danceability, energy, valence, bpm):
    bundle = models[model_choice]
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    scaler = bundle["scaler"]

    input_data = pd.DataFrame([[danceability, energy, valence, bpm]], columns=feature_names)
    input_scaled = scaler.transform(input_data)
    proba = model.predict_proba(input_scaled)[0][1]

    if DATA_AVAILABLE:
        # Top ähnliche Songs berechnen
        spotify_features = spotify_features_df[feature_names]
        distances = cdist(input_data, spotify_features, metric='euclidean')[0]
        top_indices = np.argsort(distances)[:3]
        top_songs = spotify_features_df.iloc[top_indices][['Title', 'Artist']]

        # Grammy-Status der ähnlichen Songs überprüfen
        top_songs_list = []
        for idx, row in top_songs.iterrows():
            title = row['Title']
            artist = row['Artist']
            is_grammy = ((grammy_df['song_or_album'] == title) & (grammy_df['artist'] == artist)).any()
            status = "(Grammy Winner)" if is_grammy else "(No Grammy)"
            top_songs_list.append(f"- {title.title()} von {artist.title()} {status}")

        result = f"Modell: {model_choice}\nGewinnwahrscheinlichkeit: {proba * 100:.2f}%\n\nTop ähnliche Songs:\n" + "\n".join(top_songs_list)
    else:
        result = f"Modell: {model_choice}\nGewinnwahrscheinlichkeit: {proba * 100:.2f}%\n\n⚠️ Song-Vergleich nicht verfügbar (CSV-Dateien fehlen)"
    
    return result

# Gradio UI definieren
demo = gr.Interface(
    fn=predict_grammy,
    inputs=[
        gr.Dropdown(choices=["Random Forest", "Logistic Regression"], label="Wähle ein Modell"),
        gr.Slider(0, 100, step=1, label="Danceability (0-100)"),
        gr.Slider(0, 100, step=1, label="Energy (0-100)"),
        gr.Slider(0, 100, step=1, label="Valence (Stimmung) (0-100)"),
        gr.Number(label="Tempo (Beats Per Minute)")
    ],
    outputs="text",
    title="🌟 Grammy Winning Prediction",
    description="Prognostiziere mit verschiedenen ML-Modellen, ob ein Song eine hohe Grammy-Gewinnwahrscheinlichkeit hat und finde ähnliche Top-Songs mit Grammy-Status."
)

# Starten
demo.launch(share=True)