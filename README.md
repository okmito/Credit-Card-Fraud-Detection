# Credit Card Fraud Detection System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-black?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-blueviolet?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-cyan?logo=numpy&logoColor=white)](https://numpy.org/)

</div>

A real-time fraud detection web application using a machine learning model built with **Python**, **Flask**, and **scikit-learn**. The app demonstrates training a Logistic Regression model on the popular credit card fraud dataset and serving predictions via a simple web UI.

---

## 📋 Table of Contents
- [About The Project](#about-the-project)  
- [Key Features](#key-features)  
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation & Setup](#installation--setup)  
- [How to Run](#how-to-run)  
- [Project Structure](#project-structure)  
- [Contributing](#contributing)  
- [License](#license)  
- [Contact](#contact)  

---

## 📖 About The Project
This project provides an interactive web interface to demonstrate a real-world machine learning application.  
It uses a **Logistic Regression** model to classify credit card transactions as legitimate or fraudulent using anonymized features.  
The system is designed as a practical example of deploying an ML model with Flask.

---

## ✨ Key Features
- 🤖 **Machine Learning Model**: Logistic Regression classifier (scikit-learn).  
- ⚙️ **Data Scaling**: Uses `StandardScaler` for feature normalization.  
- 🌐 **Interactive Web UI**: Flask + HTML/CSS/JavaScript front end.  
- 🎲 **Random Sample Generator**: Test the model with random, realistic data.  
- 📈 **Real-time Predictions**: Get classification + probability instantly.  

---

## 🚀 Getting Started

### Prerequisites
- Python **3.7+**
- `pip` (Python package manager)
- `git`

### Installation & Setup
```bash
# Clone the repository
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Create and activate a virtual environment
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
