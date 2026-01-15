
# Phenotype Classification Using Random Forest

## Overview
This project builds a machine learning classifier to predict cell phenotypes using numerical features derived from imaging and experimental measurements.

## Dataset
Synthetic dataset with 1000 samples, 25 features, and 3 labels:
- healthy
- stressed
- degenerating

### SHAP Explainability

SHAP (SHapley Additive exPlanations) was used to interpret the Random Forest classifier.

Because the model is multi-class (healthy / stressed / degenerating), SHAP values were computed per class and then averaged to produce a combined explanation.

This allows identification of the most influential imaging features contributing to phenotype prediction.


## How to Run
```
pip install -r requirements.txt
python src/train_model.py
```

## Technologies
Python, pandas, scikit-learn, seaborn, matplotlib
