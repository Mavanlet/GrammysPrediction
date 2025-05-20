# 🎶 Grammy-Winning Song Prediction

## Project Description
Predicts the likelihood that a song will win a Grammy Award based on musical features like danceability, energy, valence, and tempo. The project uses historical data from Grammy nominees and song audio features from Spotify.

## Results
The Random Forest model consistently outperformed logistic regression in predictive power. Still, with limited data and features, the model may not generalize to all genres or time periods. More diverse data and external validation (e.g., newer nominees) would strengthen the model’s robustness.

### Name & URL
| Name         | URL |
|--------------|-----|
| Gradio App   | [Hugging Face Space](https://huggingface.co/spaces/your-username/grammy-prediction) *(Link optional)* |
| Code         | [GitHub Repository](https://github.com/Mavanlet/GrammysPrediction) |

## Data Sources and Features Used Per Source
| Data Source | Features |
|-------------|----------|
| `grammy_winners.csv` | title, artist, won (1/0) |
| `Spotify-2000.csv`   | danceability, energy, valence, tempo, genre, popularity, duration, acousticness, speechiness |

## Features Created
| Feature | Description |
|---------|-------------|
| `danceability` | How suitable a song is for dancing (0–100 scale) |
| `energy`       | Intensity and activity level of the song |
| `valence`      | Positivity or emotional tone of the song |
| `bpm`          | Tempo in Beats per Minute |
| `scaled_features` | Features normalized with `StandardScaler` |
| `model_choice` | User can select between Logistic Regression and Random Forest |

## Model Training
### Amount of Data
- Grammy-nominated songs: 900+
- Spotify datasets: 2000

### Data Splitting Method (Train/Validation/Test)
- Data analyzed, processed, and trained in a Jupyter notebook
- Random Forest and Logistic Regression models trained with the full dataset (no explicit test set, focusing on deployment)

### Performance

| Model | Accuracy / AUC | Description |
|-------|----------------|-------------|
| Logistic Regression | ~70% AUC | Baseline model |
| Random Forest       | ~78% AUC | Higher accuracy and better feature usage |

## Demo App UI
The Gradio web app (`app.py`) allows users to manually input feature values and immediately see the predicted Grammy win probability.

- User selects a model
- Inputs values for:
  - Danceability
  - Energy
  - Valence
  - BPM
- Output: Grammy win probability in %

```bash
python app.py
```

## Requirements
```txt
pandas
scikit-learn
gradio
numpy
requests
```

## References
- [Gradio Documentation](https://gradio.app/)
- [Spotify Dataset (Kaggle)](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-160k-tracks)
- Grammy winner lists compiled manually from Wikipedia
- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)