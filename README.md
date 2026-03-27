# food-hazard-detection
Food Hazard Detection (SemEval-2025) – Two-stage NLP pipeline using TF-IDF and LinearSVC.

# Food Hazard Detection (SemEval-2025 Task 9)

This project focuses on multi-class text classification of food recall reports into:
- Hazard category (10 classes)
- Product category (22 classes)

## Approach

We implemented a two-stage classification pipeline:

1. Hazard prediction (Stage 1)
2. Product prediction using:
   - TF-IDF features
   - Predicted hazard as additional feature

## Methods

- TF-IDF (n-grams 1,2)
- LinearSVC (class_weight='balanced')
- Cross-validation to prevent data leakage
- Feature engineering (hazard-based augmentation)

## Results

- Public leaderboard score: ~0.66

## Project Structure

- pipeline.py → main training and prediction pipeline
- requirements.txt → dependencies
