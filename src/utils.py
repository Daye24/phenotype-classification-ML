import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

def load_data(path):
    """Load dataset from a CSV file."""
    return pd.read_csv(path)

def clean_data(df):
    """Fill missing values with medians."""
    return df.fillna(df.median(numeric_only=True))

def encode_labels(df, column="label"):
    """Encode categorical label column into numeric codes."""
    df[column] = df[column].astype("category").cat.codes
    return df

def split_data(df, target="label"):
    """Split DataFrame into train/test sets."""
    X = df.drop(target, axis=1)
    y = df[target]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def save_model(model, path="model.pkl"):
    """Save trained model to disk."""
    joblib.dump(model, path)

def load_model(path="model.pkl"):
    """Load model from disk."""
    return joblib.load(path)

