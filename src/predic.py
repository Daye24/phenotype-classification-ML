import pandas as pd
from utils import load_model, clean_data, encode_labels

# ---- 1. Load saved model ----
model = load_model("model.pkl")
print("Model loaded successfully.")

# ---- 2. Load new data ----
# Example: new_data.csv must have the same feature columns
df = pd.read_csv("data/new_data.csv")

# ---- 3. Prepare data ----
df = clean_data(df)

# Important: Do NOT re-encode label column; new data should not contain labels
# df = encode_labels(df)  # <-- only for training data

# ---- 4. Make predictions ----
preds = model.predict(df)

# ---- 5. Map numeric classes back to labels
label_map = {0: "healthy", 1: "stressed", 2: "degenerating"}
decoded_preds = [label_map[p] for p in preds]

# ---- 6. Output predictions ----
print("\nPredictions:")
print(decoded_preds)
