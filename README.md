Credit Card Fraud Detection System
<div align="center">










</div>

A real-time fraud detection web application using a machine learning model built with Python, Flask, and scikit-learn. The app demonstrates training a Logistic Regression model on the popular credit-card fraud dataset and serving predictions via a simple web UI.

Table of Contents

About The Project

Key Features

Getting Started

Prerequisites

Installation & Setup

How to Run

Train the model

Start the web app

Project Structure

Example Files / Notes

Contributing

License

Contact

About The Project

This project provides an interactive web interface to demonstrate a real-world machine learning application. It uses a Logistic Regression model to classify credit card transactions as legitimate or fraudulent using anonymized features. The system is a practical example of deploying an ML model with Flask.

Key Features

Machine Learning Model: Logistic Regression classifier (scikit-learn).

Data Scaling: Uses StandardScaler for feature normalization.

Interactive Web UI: Minimal UI implemented with Flask + HTML/CSS/JS.

Random Sample Generator: Button to generate randomized realistic samples for testing.

Real-time Predictions: Returns a prediction and a confidence score instantly.

Getting Started
Prerequisites

Python 3.7+

pip (Python package manager)

git (optional, but recommended)

Installation & Setup
# clone repo
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# create & activate virtual environment (macOS / Linux)
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# install requirements
pip install -r requirements.txt


requirements.txt (example minimal contents)

Flask>=2.0
scikit-learn>=1.0
pandas>=1.3
numpy>=1.21
joblib
gunicorn   # optional for production

Dataset (Kaggle)

This project uses the "Credit Card Fraud Detection" dataset from Kaggle.

Kaggle dataset page: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download creditcard.csv from the Kaggle dataset page.

Place creditcard.csv in the project root (or update the path in model_training.py if you put it elsewhere).

If you don't already have a Kaggle account or the API set up, download manually from the web interface.

How to Run
Train the model

Run the training script from the project root. It trains the model and saves it to model/ (example):

python model_training.py


Expect this script to:

Load creditcard.csv

Preprocess (scaling, optional resampling)

Train a LogisticRegression model

Save the trained model and scaler (e.g. using joblib.dump) into model/:

model/logreg_model.joblib

model/scaler.joblib

(See Example Files / Notes below for a minimal pattern.)

Start the web app

After training, run the Flask app:

# Option 1: simple direct run
python app.py

# Option 2: using flask CLI
export FLASK_APP=app.py      # macOS / Linux
set FLASK_APP=app.py         # Windows (cmd)
flask run --host=127.0.0.1 --port=5000

# Option 3: production (gunicorn)
gunicorn -w 4 app:app


Open your browser to:

http://127.0.0.1:5000

Project Structure (suggested)
credit-card-fraud-detection/
├─ app.py                  # Flask application
├─ model_training.py       # Training script
├─ requirements.txt
├─ README.md
├─ creditcard.csv          # (downloaded dataset)
├─ model/
│  ├─ logreg_model.joblib
│  └─ scaler.joblib
├─ templates/
│  └─ index.html
├─ static/
│  ├─ css/
│  └─ js/
└─ LICENSE

Example Files / Notes
model_training.py (recommended behavior)

Load dataset (creditcard.csv) with pandas.

Feature split (X / y).

Fit StandardScaler() to numeric features and transform.

Train LogisticRegression() (set max_iter suitably).

Save scaler and model using joblib.dump into model/.

Minimal pseudo-outline:

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from joblib import dump

df = pd.read_csv('creditcard.csv')
X = df.drop('Class', axis=1)
y = df['Class']

# scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
clf = LogisticRegression(max_iter=1000, class_weight='balanced')
clf.fit(X_train, y_train)

# save
os.makedirs('model', exist_ok=True)
dump(clf, 'model/logreg_model.joblib')
dump(scaler, 'model/scaler.joblib')
print("Saved model and scaler to /model")

app.py (recommended behavior)

Load model/logreg_model.joblib and model/scaler.joblib at startup.

Provide a web UI with a form or an API endpoint /predict that accepts a JSON payload or form fields, scales the input, predicts, and returns prediction + probability.

Minimal outline:

from flask import Flask, request, render_template, jsonify
from joblib import load
import numpy as np

app = Flask(__name__)
model = load('model/logreg_model.joblib')
scaler = load('model/scaler.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # or request.form
    # convert to numpy array with right order of features
    X = np.array([data['feature1'], data['feature2'], ...]).reshape(1, -1)
    Xs = scaler.transform(X)
    prob = model.predict_proba(Xs)[0,1]  # probability of fraud
    pred = int(model.predict(Xs)[0])
    return jsonify({'prediction': pred, 'fraud_probability': float(prob)})


Important: Ensure feature order used in app matches the one used during training.

Contributing

Contributions are welcome! Typical workflow:

Fork the repository

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request