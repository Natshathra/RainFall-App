# model_train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
data = pd.read_csv('data/weatherAUS.csv')

# Drop rows with missing target
data.dropna(subset=['RainTomorrow'], inplace=True)

# Fill missing values
data.fillna(method='ffill', inplace=True)

# Encode categorical features
label_encoders = {}
categorical_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Encode target
data['RainTomorrow'] = data['RainTomorrow'].map({'Yes': 1, 'No': 0})

# Features & target
X = data[['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Humidity9am', 'Humidity3pm', 'RainToday']]
y = data['RainTomorrow']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
with open('models/rain_model.pkl', 'wb') as f:
    pickle.dump(model, f)
