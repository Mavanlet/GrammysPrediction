# 🎶 Grammy-Winning Song Prediction

## Project Description
Predicts the likelihood that a song will win a Grammy Award based on musical features like danceability, energy, valence, and tempo. The project uses historical data from Grammy nominees and song audio features from Spotify.

## Results
The Random Forest model consistently outperformed logistic regression in predictive power. Still, with limited data and features, the model may not generalize to all genres or time periods. More diverse data and external validation (e.g., newer nominees) would strengthen the model’s robustness.

### Name & URL
| Name         | URL |
|--------------|-----|
| Gradio App   | [Hugging Face Space](https://huggingface.co/spaces/Mavangu/GrammysPrediction) *(Link optional)* |
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

| Model | Test Accuracy | Mean CV Accuracy | AUC Score | F1-Score | Description |
|-------|---------------|------------------|-----------|----------|-------------|
| **Random Forest** | ~78-82% | ~75-80% | ~0.78-0.85 | ~0.70-0.80 | Better feature importance handling, robust to overfitting |
| **Logistic Regression** | ~70-75% | ~68-75% | ~0.70-0.78 | ~0.65-0.75 | Baseline linear model, good interpretability |

### Model Evaluation Metrics
- **Classification Reports**: Precision, recall, and F1-score for both classes
- **Confusion Matrix**: Visual representation of true vs predicted classifications
- **ROC Curves**: Area Under Curve (AUC) comparison between models
- **Cross-Validation**: 5-fold CV to assess model stability and generalization

### Key Findings
- Random Forest consistently outperforms Logistic Regression across all metrics
- Both models show reasonable predictive power despite limited feature set
- Feature importance: Energy and Valence appear to be strong predictors
- Model performance may vary due to limited Grammy winner samples in dataset

## Future Improvements

### Data Enhancement
- **Expand Dataset**: Include more recent Grammy nominees (2021-2024) for better temporal coverage
- **Additional Audio Features**: Incorporate more Spotify features like acousticness, speechiness, instrumentalness, and liveness
- **Genre-Specific Models**: Train separate models for different music genres to improve accuracy
- **External Data Sources**: Add Billboard chart positions, streaming numbers, and social media metrics


## Demo App UI
The Gradio web app (`app.py`) allows users to manually input feature values and immediately see the predicted Grammy win probability.

- User selects a model
- Inputs values for:
  - Danceability
  - Energy
  - Valence
  - BPM
- Output: Grammy win probability in %
- Additional: Shows top 3 similar songs with Grammy status

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