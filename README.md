# Interpretable PRO-based Pain Prediction: Implementation of Attention-RNN


## Overview

This repository contains the implementation of the model described in the paper **"Recurrent neural networks and attention scores for personalized prediction and interpretation of patient-reported outcomes"**, published in the *Journal of Biopharmaceutical Statistics*. The model utilizes **Recurrent Neural Networks (RNNs)** combined with **attention scores** to predict and interpret Patient-Reported Outcomes (PROs) such as pain interference, fatigue, and sleep disturbance, derived from longitudinal clinical trial data.

**Paper URL:** [Recurrent neural networks and attention scores for personalized prediction and interpretation of patient-reported outcomes](https://www.tandfonline.com/doi/abs/10.1080/10543406.2025.2469884)


## Key Contributions
- **Personalized Predictions:** The model uses RNNs to handle longitudinal data and predict outcomes at individual time points for patients.
- **Interpretability:** Attention scores are integrated into the RNN framework to highlight the most relevant input features (e.g., PRO items, treatment types) that contribute to the prediction at both the individual and group levels.
- **Clinical Decision Support:** The model aims to assist in clinical decision-making by providing insights into treatment effectiveness and symptom management based on patient history.

## Methodology

- **Dataset:** The model uses data from a randomized clinical trial involving 402 patients with Cryptogenic Sensory Polyneuropathy (CSPN). The dataset includes PROs like SF-12, PROMIS pain interference, fatigue, and sleep disturbance.
- **Model Architecture:** The Attention-RNN model incorporates two sets of attention mechanisms:
  - **Time-level attention**: Weighs the importance of different time points (e.g., baseline, week 4, week 8, and week 12).
  - **Feature-level attention**: Focuses on PRO items and treatment drugs.
- **Performance Metrics:** The model is evaluated using **Mean Absolute Error (MAE)** and **R-squared (R²)** statistics, demonstrating a high level of predictive performance.

## Results
- **Prediction Performance**: The Attention-RNN model achieved a Mean Absolute Error (MAE) of 3.7 and an R² of 0.63 on predicting pain interference scores.

- **Feature Importance**: The attention scores provide insights into the most significant PRO items and treatment drugs influencing the model's predictions. For example, Mexiletine emerged as the most influential treatment at various time points, and specific PRO items related to fatigue and social activity interference were crucial predictors.
