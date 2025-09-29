<div align="center">
<h1>Credit Card Fraud Detection System</h1>
<p>
A real-time fraud detection application using a machine learning model built with Python, Flask, and Scikit-learn.
</p>

<p>
<img alt="Python" src="https://www.citypng.com/public/uploads/preview/hd-python-logo-symbol-transparent-png-735811696257415dbkifcuokn.png">
<img alt="Flask" src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQzxkF5yUZ387TY1acpUQYU7Du1yYug1et7Hw&s">
<img alt="Scikit-learn" src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQMtySuHgewygE0E3Cpz4ugYZY9VT8B2iDiug&s">
</p>
</div>

📜 Table of Contents
About The Project

Key Features

Getting Started

Prerequisites

Installation & Setup

How to Run

Project Structure

Contributing

License

Contact

📋 About The Project
This project provides an interactive web interface to demonstrate a real-world machine learning application. It uses a Logistic Regression model to classify credit card transactions as either legitimate or fraudulent based on a set of anonymized features. The system is designed to be a practical example of deploying an ML model with a web framework.

✨ Key Features
🤖 Machine Learning Model: Uses a Logistic Regression model trained to distinguish between legitimate and fraudulent transactions.

⚙️ Data Scaling: Implements StandardScaler to normalize input features for better model performance.

🌐 Interactive Web Interface: A clean and user-friendly UI built with Flask and HTML/CSS/JavaScript.

🎲 Dynamic Data Generation: "Generate Random Sample" button to quickly test the model with randomized, realistic data.

📈 Real-time Predictions: Instantly get a prediction and a confidence score for the transaction's legitimacy.

🚀 Getting Started
Follow these instructions to set up and run the project on your local machine.

1. Prerequisites
Python 3.7+

pip (Python package installer)

Git

2. Installation & Setup
Clone the repository:

git clone [https://github.com/your-username/credit-card-fraud-detection.git](https://github.com/your-username/credit-card-fraud-detection.git)
cd credit-card-fraud-detection

Create and activate a virtual environment:

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

Install the required libraries:

pip install -r requirements.txt

Download the dataset:
The model is trained on the "Credit Card Fraud Detection" dataset from Kaggle.

Download it from this link.

Place the downloaded creditcard.csv file in the root directory of the project.

▶️ How to Run
Train the model:
Run the training script from the root directory. This will create a model folder and save the trained model inside it.

python model_training.py

Start the web application:
Once the model is trained, start the Flask web server.

python app.py

Access the application:
Open your web browser and navigate to 👉 https://www.google.com/search?q=http://127.0.0.1:5000

📁 Project Structure
credit-card-fraud-detection/
│
├── model/
│   └── fraud_detection_model.pkl
│
├── templates/
│   └── index.html
│
├── .gitignore
├── app.py
├── model_training.py
├── requirements.txt
└── README.md

🤝 Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)
    
Open a Pull Request
