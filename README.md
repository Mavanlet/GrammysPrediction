# 🎶 Grammy-Winning Song Prediction

## Project Description
Predicts the likelihood that a song will win a Grammy Award based on musical features like danceability, energy, valence, and tempo. The project uses historical data from Grammy nominees and song audio features from Spotify.

## Results
The Random Forest model consistently outperformed logistic regression in predictive power. Still, with limited data and features, the model may not generalize to all genres or time periods. More diverse data and external validation (e.g., newer nominees) would strengthen the model’s robustness.

### Class Imbalance Impact

Of the 1,994 songs in the Spotify dataset, only 185 are labeled as Grammy winners (~9.3%). This significant class imbalance makes the classification task challenging. Both models, especially Logistic Regression, struggle to correctly identify the positive class, leading to low recall and precision for Grammy-winning songs. This highlights a key limitation: the model currently performs well on the dominant class (non-winners), but underperforms on the minority class. 


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
- The dataset was split using `train_test_split` (80% training, 20% testing) to evaluate model performance on unseen data.
- Both models (Random Forest and Logistic Regression) were trained on the training set and tested on the holdout test set.
- In addition, 5-fold cross-validation was performed on the training set to assess model stability and generalization.

### Performance

| Model               | Test Accuracy | Mean CV Accuracy | AUC Score | F1-Score | Description |
|---------------------|---------------|------------------|-----------|----------|-------------|
| **Random Forest**        | 0.62          | 0.61             | 0.59      | 0.67     | Balanced model that is less sensitive to outliers and irregular data. However, in this dataset it showed lower consistency and weaker separation power (AUC). |
| **Logistic Regression**  | 0.64          | 0.56             | 0.62      | 0.73     | Simple linear model that performed better on this test set, especially in recall and F1-score. Less stable across cross-validation folds. |


### Model Evaluation Metrics
- **Accuracy**: Percentage of correctly predicted Grammy winners in the test set
- **F1-Score**: Harmonic mean of precision and recall, balancing false positives and negatives
- **Confusion Matrix**: Used to derive true positives, false positives, and overall accuracy
- **ROC Curves and AUC Score**: AUC was relatively low (~0.59–0.62), indicating weak separation between classes
- **Cross-Validation**: 5-fold cross-validation applied to estimate model generalization, with noticeable variation in some folds


### Key Findings
- Logistic Regression outperformed Random Forest in test accuracy, F1-score, and AUC on this dataset
- Both models show limited but usable predictive performance, constrained by the small feature set and dataset size
- Features such as Energy and Valence appear to be the most influential in determining Grammy win likelihood
- Model performance likely suffers from imbalanced or limited Grammy-winning song data


## Future Improvements

### Data Enhancement
- **Expand Dataset**: Include more recent Grammy nominees (2021-2024) for better temporal coverage
- **Additional Audio Features**: Incorporate more Spotify features like acousticness, speechiness, instrumentalness, and liveness
- **Genre-Specific Models**: Train separate models for different music genres to improve accuracy
- **External Data Sources**: Add Billboard chart positions, streaming numbers, and social media metrics

**Direct Audio File Upload**: Allow users to upload MP3/WAV files for direct analysis
- **Audio Feature Extraction**: Implement librosa or Spotify Web API to extract audio features from uploaded files
- **Real-Time Audio Processing**: Process audio files in real-time to extract danceability, energy, valence, and tempo
- **Waveform Visualization**: Display audio waveforms and spectrograms alongside predictions
- **Multi-Format Support**: Support various audio formats (MP3, WAV, FLAC, M4A)

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