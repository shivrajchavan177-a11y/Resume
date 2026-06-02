import pandas as pd
import numpy as np
import pickle
import ast
import re

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

print("=" * 60)
print("AI Resume Screening Model Training")
print("=" * 60)

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

df = pd.read_csv("resume_data.csv")

print(f"\nDataset Shape: {df.shape}")

# ---------------------------------------------------
# SELECT IMPORTANT COLUMNS
# ---------------------------------------------------

required_columns = [
    'skills',
    'matched_score',
    '﻿job_position_name'
]

df = df[required_columns]

# rename weird column name
df.rename(columns={
    '﻿job_position_name': 'job_role'
}, inplace=True)

# remove missing rows
df.dropna(subset=['skills', 'matched_score', 'job_role'], inplace=True)

print(f"After Cleaning: {df.shape}")

# ---------------------------------------------------
# CLEAN SKILLS
# ---------------------------------------------------

def clean_skills(skill_text):

    try:
        # convert string list to actual list
        skills = ast.literal_eval(skill_text)

    except:
        skills = []

    cleaned = []

    for skill in skills:

        skill = str(skill).lower().strip()

        # remove special characters
        skill = re.sub(r'[^a-zA-Z0-9#+ ]', '', skill)

        if len(skill) > 1:
            cleaned.append(skill)

    return list(set(cleaned))


df['skills'] = df['skills'].apply(clean_skills)

# remove empty skills rows
df = df[df['skills'].map(len) > 0]

print(f"After Skill Cleaning: {df.shape}")

# ---------------------------------------------------
# ENCODE SKILLS
# ---------------------------------------------------

mlb = MultiLabelBinarizer()

skills_encoded = mlb.fit_transform(df['skills'])

print(f"\nTotal Unique Skills: {len(mlb.classes_)}")

# ---------------------------------------------------
# PREPARE FEATURES
# ---------------------------------------------------

X = skills_encoded

# convert score from 0-1 to 1-10 scale
y = df['matched_score'] * 10

# ---------------------------------------------------
# SPLIT DATA
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------

print("\nTraining Random Forest Model...")

model = RandomForestRegressor(
    n_estimators=150,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model Training Completed!")

# ---------------------------------------------------
# EVALUATION
# ---------------------------------------------------

predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"R2 Score      : {round(r2, 3)}")
print(f"MAE Error     : {round(mae, 3)}")

accuracy = max(0, min(100, r2 * 100))

print(f"Approx Accuracy: {round(accuracy, 2)}%")

# ---------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(mlb, open("mlb.pkl", "wb"))

print("\nFiles Saved Successfully!")
print("✔ model.pkl")
print("✔ mlb.pkl")

print("\nTraining Completed Successfully!")
