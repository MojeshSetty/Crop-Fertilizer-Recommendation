import numpy as np
import pandas as pd
import xgboost as xgb
import sqlite3

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Connect or create SQLite DB and table
conn = sqlite3.connect('crop.db')
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS crop_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    N INTEGER, P INTEGER, K INTEGER,
    temperature REAL, humidity REAL, ph REAL, rainfall REAL, label TEXT
)
""")
conn.commit()

# Load from database
crop = pd.read_sql_query("SELECT N, P, K, temperature, humidity, ph, rainfall, label FROM crop_data", conn)

# If DB is empty, uncomment for initial population:
# df = pd.read_csv("Croprecommendation.csv")
# df.to_sql("crop_data", conn, if_exists="replace", index=False)
# crop = df.copy()

# Preprocessing/ML pipeline
crop['label'] = crop['label'].astype('category')
y = crop['label'].cat.codes
X = crop.drop('label', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2347)

model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(np.unique(y)), learning_rate=0.01, max_depth=6, n_estimators=100, seed=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(f"Accuracy is {accuracy_score(y_test, y_pred) * 100:.2f}")

# Prediction + DB insert
def croprecommendation(N, P, K, temp, hum, ph_val, rain):
    features = np.array([N, P, K, temp, hum, ph_val, rain])
    prediction = model.predict(features.reshape(1, -1))[0]
    crop_dict = dict(enumerate(crop['label'].cat.categories))
    predicted_crop = crop_dict[prediction]
    cursor.execute("""
        INSERT INTO crop_data (N, P, K, temperature, humidity, ph, rainfall, label)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (N, P, K, temp, hum, ph_val, rain, predicted_crop))
    conn.commit()
    return f"{predicted_crop} is the best crop to grow in the farm."

print(croprecommendation(90, 45, 43, 20, 82, 6.1, 202))

cursor.close()
conn.close()
