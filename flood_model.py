# ==========================================================
# RISING WATERS - FLOOD PREDICTION SYSTEM
# Flood_prediction.py
# ==========================================================

# ==========================
# IMPORT LIBRARIES
# ==========================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Create models folder automatically
os.makedirs("models", exist_ok=True)

# ==========================================================
# EPIC 1 - DATA COLLECTION
# ==========================================================

print("=" * 60)
print("EPIC 1 - DATA COLLECTION")
print("=" * 60)

df = pd.read_excel("dataset/flood dataset.xlsx")

print("\nFirst 5 Records")
print(df.head())

print("\nLast 5 Records")
print(df.tail())

print("\nDataset Shape")
print(df.shape)

print("\nColumn Names")
print(df.columns.tolist())

print("\nDataset Information")
df.info()

print("\nStatistical Summary")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows")
print(df.duplicated().sum())

# ==========================================================
# EPIC 2 - DATA VISUALIZATION
# ==========================================================

print("\n" + "=" * 60)
print("EPIC 2 - DATA VISUALIZATION")
print("=" * 60)

# Histogram
print("\nDisplaying Histograms...")

df.hist(figsize=(16,12))
plt.suptitle("Histogram of Dataset Features")
plt.tight_layout()
plt.show()

# Correlation Heatmap

plt.figure(figsize=(12,8))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    linewidths=0.5
)
plt.title("Correlation Heatmap")
plt.show()

# Box Plot

plt.figure(figsize=(15,6))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title("Box Plot")
plt.show()

# Flood Distribution

plt.figure(figsize=(6,4))
sns.countplot(x="flood", data=df)
plt.title("Flood Distribution")
plt.xlabel("Flood")
plt.ylabel("Count")
plt.show()

# ==========================================================
# EPIC 3 - DATA PREPROCESSING
# ==========================================================

print("\n" + "=" * 60)
print("EPIC 3 - DATA PREPROCESSING")
print("=" * 60)

# Missing values

df.fillna(df.mean(numeric_only=True), inplace=True)

# Remove duplicates

df.drop_duplicates(inplace=True)

print("\nDataset Shape After Cleaning")
print(df.shape)

# Encode categorical columns if present

categorical_columns = df.select_dtypes(include=["object"]).columns

for column in categorical_columns:
    df[column] = df[column].astype("category").cat.codes

# Separate Features and Target

X = df.drop("flood", axis=1)
y = df["flood"]

print("\nFlood Class Distribution")
print(y.value_counts())

# Save feature names
feature_columns = X.columns.tolist()
print("\nInput Features")

for i, feature in enumerate(feature_columns, start=1):
    print(f"{i}. {feature}")
    
    
print("\nTarget Column")
print("flood")

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

# Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeature Scaling Completed Successfully.")   


# ==========================================================
# EPIC 4 - MODEL BUILDING
# ==========================================================

print("\n" + "=" * 60)
print("EPIC 4 - MODEL BUILDING")
print("=" * 60)

# -------------------------
# Decision Tree
# -------------------------

print("\nTraining Decision Tree...")

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

pred_dt = dt.predict(X_test)

dt_accuracy = accuracy_score(y_test, pred_dt)

print("Decision Tree Accuracy :", f"{dt_accuracy*100:.2f}%")

# -------------------------
# Random Forest
# -------------------------

print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=3,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)
rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, pred_rf)

print("Random Forest Accuracy :", f"{rf_accuracy*100:.2f}%")

# -------------------------
# KNN
# -------------------------

print("\nTraining KNN...")

knn = KNeighborsClassifier(
    n_neighbors=5
)

knn.fit(X_train, y_train)

pred_knn = knn.predict(X_test)

knn_accuracy = accuracy_score(y_test, pred_knn)

print("KNN Accuracy :", f"{knn_accuracy*100:.2f}%")

# -------------------------
# XGBoost
# -------------------------

print("\nTraining XGBoost...")

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)
xgb.fit(X_train, y_train)

pred_xgb = xgb.predict(X_test)

xgb_accuracy = accuracy_score(y_test, pred_xgb)

print("XGBoost Accuracy :", f"{xgb_accuracy*100:.2f}%")

# ==========================================================
# MODEL COMPARISON
# ==========================================================

models = {
    "Decision Tree": (dt, dt_accuracy),
    "Random Forest": (rf, rf_accuracy),
    "KNN": (knn, knn_accuracy),
    "XGBoost": (xgb, xgb_accuracy)
}

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

for model_name, (_, acc) in models.items():
    print(f"{model_name:<20}: {acc*100:.2f}%")

# ==========================================================
# BEST MODEL
# ==========================================================

best_model_name = max(models, key=lambda x: models[x][1])

best_model = models[best_model_name][0]

best_accuracy = models[best_model_name][1]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Best Model :", best_model_name)
print("Accuracy   :", f"{best_accuracy*100:.2f}%")

# ==========================================================
# MODEL EVALUATION
# ==========================================================

best_prediction = best_model.predict(X_test)
print("\nTrain Accuracy :", f"{best_model.score(X_train, y_train)*100:.2f}%")
print("Test Accuracy  :", f"{best_model.score(X_test, y_test)*100:.2f}%")

print("\nClassification Report")

print(
    classification_report(
        y_test,
        best_prediction,
        zero_division=0
    )
)

print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        best_prediction
    )
)

# ==========================================================
# EPIC 5 - SAVE MODEL
# ==========================================================

print("\n" + "=" * 60)
print("EPIC 5 - SAVING MODEL")
print("=" * 60)

joblib.dump(
    best_model,
    "models/flood_prediction_model.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

joblib.dump(
    feature_columns,
    "models/feature_columns.pkl"
)

model_info = {
    "best_model": best_model_name,
    "accuracy": best_accuracy
}

joblib.dump(
    model_info,
    "models/model_info.pkl"
)

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"Best Model : {best_model_name}")
print(f"Accuracy   : {best_accuracy*100:.2f}%")

print("\nSaved Files")

print("- flood_prediction_model.pkl")
print("- scaler.pkl")
print("- feature_columns.pkl")
print("- model_info.pkl")

print("\nYour Flask Application is Ready.")

print("=" * 60)