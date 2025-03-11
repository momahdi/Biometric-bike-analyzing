import json
import os
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# Define directory and load lap files
lap_directory = "./Segmented/P01"
lap_files = sorted(glob.glob(f"{lap_directory}/lap_*.json"))

# Ensure we have at least 5 laps
if len(lap_files) < 5:
    raise ValueError("Not enough laps available. At least 5 laps are required.")

# Load each lap separately
laps = []
for lap_file in lap_files:
    with open(lap_file, "r") as file:
        laps.append(json.load(file))

# Extract Brake Data and Time
def extract_brake_data(lap_data):
    brake_data = []
    for entry in lap_data:
        brake_value = float(entry["brakeData"])
        timestamp = float(entry["unixTimeStamp"])
        participant_id = entry.get("PID", "P01")  
        brake_data.append([timestamp, brake_value, participant_id])
    return pd.DataFrame(brake_data, columns=["timestamp", "brake_value", "participant_id"])

# Use laps 1-4 for training and lap 5 for testing
train_laps = laps[:4]
test_lap = laps[4]

# Convert to DataFrame
df_train = pd.concat([extract_brake_data(lap) for lap in train_laps], ignore_index=True)
df_test = extract_brake_data(test_lap)

# Encode participant IDs
df_train["participant_id"] = df_train["participant_id"].astype("category").cat.codes
df_test["participant_id"] = df_test["participant_id"].astype("category").cat.codes

# Define features (X) and labels (y)
X_train = df_train[["timestamp", "brake_value"]]
y_train = df_train["participant_id"]
X_test = df_test[["timestamp", "brake_value"]]
y_test = df_test["participant_id"]

# Initialize models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
svm_model = SVC(kernel='linear', random_state=42)

# Train models
rf_model.fit(X_train, y_train)
# svm_model.fit(X_train, y_train)

# Predict
rf_preds = rf_model.predict(X_test)
# svm_preds = svm_model.predict(X_test)

# Evaluate models
rf_accuracy = accuracy_score(y_test, rf_preds)
rf_f1 = f1_score(y_test, rf_preds, average='weighted')
# svm_accuracy = accuracy_score(y_test, svm_preds)
# svm_f1 = f1_score(y_test, svm_preds, average='weighted')

# Print results
print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest F1 Score:", rf_f1)
# print("SVM Accuracy:", svm_accuracy)
# print("SVM F1 Score:", svm_f1)

# CURRENT RESULT
# Random Forest Accuracy: 1.0
# Random Forest F1 Score: 1.0
# Note segmented is not correct