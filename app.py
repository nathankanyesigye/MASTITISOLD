from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

app = Flask(__name__)

# Load the trained XGBoost model
model = xgb.XGBClassifier()
model.load_model('xgboost_model_v4.json')

# Initialize preprocessors
le = LabelEncoder()
scaler = StandardScaler()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read CSV
        df = pd.read_csv(file)

        # Preprocess data
        if 'class1' not in df.columns:
            return jsonify({"error": "CSV must contain 'class1' column for true labels"}), 400

        X = df.drop('class1', axis=1)
        y = df['class1']

        # Encode categorical variables safely
        for col in ['Cow_ID', 'Breed']:
            if col in X.columns:
                X[col] = le.fit_transform(X[col])

        # Scale numerical features
        numerical_columns = X.select_dtypes(include=['float64', 'int64']).columns
        X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

        # Predictions
        y_pred = model.predict(X)

        # Return predictions as JSON
        return jsonify({"predictions": y_pred.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
