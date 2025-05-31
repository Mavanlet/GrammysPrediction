# 🎶 Grammy-Winning Song Prediction

## Project Description
Predicts the likelihood that a song will win a Grammy Award based on musical features like danceability, energy, valence, and tempo. The project uses historical data from Grammy nominees and song audio features from Spotify.

## Results
Logistic Regression outperformed Random Forest on this dataset, achieving 64% test accuracy and 83% Grammy winner recall. Both models show underfitting due to limited features (4) and dataset size (249 songs). The low AUC scores (0.59-0.62) indicate that Grammy prediction requires more comprehensive data beyond basic audio features.

### Name & URL
| Name         | URL |
|--------------|-----|
| Gradio App   | [Hugging Face Space](https://huggingface.co/spaces/Mavangu/GrammysPrediction) |
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
- **Matched songs used for training:** 249 (Grammy + Spotify data)
- **Train/Test split:** 199 training, 50 testing

### Data Limitations
- Dataset limited to 249 songs where Grammy nominees could be exactly matched with Spotify audio features
- Strict string matching may miss some potential matches due to naming variations
- Model trained on successfully matched subset, representing a sample of Grammy-nominated songs

### Data Splitting Method (Train/Validation/Test)
- The dataset was split using `train_test_split` (80% training, 20% testing) to evaluate model performance on unseen data.
- Both models (Random Forest and Logistic Regression) were trained on the training set and tested on the holdout test set.
- In addition, 5-fold cross-validation was performed on the training set to assess model stability and generalization.

### Performance

| Model | Test Accuracy | Mean CV Accuracy | AUC Score | F1-Score (Grammy) | Grammy Recall |
|-------|---------------|------------------|-----------|-------------------|---------------|
| **Random Forest** | 62.0% | 61.3% | 0.594 | 0.667 | 66% |
| **Logistic Regression** | **64.0%** | 56.2% | **0.621** | **0.727** | **83%** |

#### Cross-Validation Details
- **Random Forest CV scores:** [0.65, 0.50, 0.65, 0.70, 0.56] → Mean: 61.3%
- **Logistic Regression CV scores:** [0.55, 0.60, 0.65, 0.575, 0.44] → Mean: 56.2%

#### Test Set Composition
- **50 test songs:** 21 non-Grammy winners, 29 Grammy winners
- **Class distribution:** 58% Grammy winners, 42% non-winners in test set

### Model Evaluation Metrics
- **Accuracy**: Percentage of correctly predicted Grammy winners in the test set
- **F1-Score**: Harmonic mean of precision and recall, balancing false positives and negatives
- **Confusion Matrix**: Used to derive true positives, false positives, and overall accuracy
- **ROC Curves and AUC Score**: AUC was relatively low (~0.59–0.62), indicating weak separation between classes
- **Cross-Validation**: 5-fold cross-validation applied to estimate model generalization, with noticeable variation in some folds


### Key Findings

#### **Winner: Logistic Regression**
- **17% better Grammy detection:** 83% vs 66% recall for identifying actual Grammy winners
- **Higher overall accuracy:** 64% vs 62% on unseen test data
- **Better class discrimination:** AUC 0.621 vs 0.594
- **Superior generalization:** Better performance on new data despite lower CV scores

#### **Model Limitations (Underfitting)**
Both models show underfitting characteristics:
- **Limited feature set:** Only 4 audio features insufficient for complex Grammy decisions
- **Low AUC scores:** 0.59-0.62 indicates poor class separation (barely better than random guessing)
- **Complex prediction task:** Grammy success depends on factors beyond audio characteristics
- **Small dataset:** 249 songs may be insufficient for capturing full Grammy prediction complexity

#### **Why Logistic Regression Outperforms Random Forest**
- **Optimal for small datasets:** Less prone to overfitting with limited data (249 songs)
- **Simpler model advantage:** Finds generalizable patterns rather than memorizing training specifics
- **Better generalization:** CV 56.2% → Test 64% (+7.8%) vs RF's CV 61.3% → Test 62% (+0.7%)
- **Linear relationship detection:** Effectively captures audio feature correlations for Grammy prediction

*Note: Random Forest would likely perform better with 1000+ songs and 10+ features.*


## Future Improvements

### Model Enhancement
- **XGBoost/LightGBM:** Better performance with limited data and feature interactions
- **Ensemble Methods:** Combine multiple models for improved accuracy
- **Feature Engineering:** Create interaction features (e.g., Energy × Danceability)
- **Hyperparameter Tuning:** Optimize model parameters using GridSearch/RandomSearch

#### Advanced Models (Requires Larger Dataset: 10,000+ songs)
- **Neural Networks:** Deep learning approaches need significantly more data and features
  - **Convolutional Neural Networks (CNN):** For raw audio spectrogram analysis
  - **Recurrent Neural Networks (LSTM):** For temporal audio sequence patterns  
  - **Transformer Models:** For multi-modal analysis (audio + lyrics + metadata)
- **Deep Ensemble:** Combining multiple neural network architectures

*Note: Current dataset (249 songs, 4 features) is insufficient for deep learning approaches. Neural networks would severely overfit and perform worse than classical ML models.*

### Data Enhancement
- **Expand Dataset**: Include more recent Grammy nominees (2021-2024) for better temporal coverage
- **Additional Audio Features**: Incorporate more Spotify features like acousticness, speechiness, instrumentalness, and liveness
- **External Data Sources**: Add Billboard chart positions, streaming numbers, and social media metrics

### Advanced Features
- **Genre-Specific Models:** Train separate models for different music genres to improve accuracy
- **Temporal Analysis:** Include Grammy year trends and decade-specific patterns
- **Artist Features:** Add artist popularity, previous Grammy wins, label size
- **Market Data:** Billboard chart positions, streaming numbers, social media metrics
- **Audio Complexity:** Add more Spotify features (acousticness, speechiness, instrumentalness, liveness)

### App Enhancement Features
- **Direct Audio File Upload**: Allow users to upload MP3/WAV files for direct analysis
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
- [Grammy Winners and Nominees from 1965 to 2024 (Kaggle)](https://www.kaggle.com/datasets/johnpendenque/grammy-winners-and-nominees-from-1965-to-2024)
- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)