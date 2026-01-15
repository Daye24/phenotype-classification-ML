import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from utils import load_data, clean_data, encode_labels, split_data, save_model

# ---- 1. Load and prepare data ----
df = load_data("data/dataset.csv")
df = clean_data(df)
df = encode_labels(df, "label")

# ---- 2. Train-test split ----
X_train, X_test, y_train, y_test = split_data(df)

# ---- 3. Train model ----
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# ---- 4. Evaluate model ----
preds = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:\n", classification_report(y_test, preds))

# ---- 5. Confusion matrix ----
cm = confusion_matrix(y_test, preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# ---- 6. Save model ----
save_model(rf, "model.pkl")
print("\nModel saved as model.pkl")

