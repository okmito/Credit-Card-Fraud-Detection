import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler # Import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import os

# --- Step 1: Load The Data ---
print("Loading dataset...")
try:
    data = pd.read_csv('creditcard.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'creditcard.csv' not found. Please make sure the dataset file is in the correct directory.")
    exit()


# --- Step 2: Prepare The Data ---
print("Preparing data for training...")
features_to_keep = ['V1', 'V2', 'V3', 'V4', 'V5', 'Amount']
target = 'Class'
print(f"Using a simplified model with {len(features_to_keep)} features.")
X = data[features_to_keep]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data preparation complete.")


# --- Step 3: Scale The Features ---
print("Scaling numerical features...")
# Initialize the StandardScaler
scaler = StandardScaler()
# Fit the scaler ONLY on the training data to avoid data leakage
X_train_scaled = scaler.fit_transform(X_train)
# Transform the test data using the scaler fitted on the training data
X_test_scaled = scaler.transform(X_test)
print("Feature scaling complete.")


# --- Step 4: Train The Model ---
print("Training the Logistic Regression model...")
# We remove the C parameter to let the model be more sensitive, and scaling will help manage probabilities.
model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')
# Train the model on the SCALED training data
model.fit(X_train_scaled, y_train)
print("Model training complete.")


# --- Step 5: Evaluate The Model ---
print("\n--- Model Evaluation ---")
# Evaluate on the SCALED test data
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("-------------------------\n")


# --- Step 6: Save The Model and Scaler ---
model_dir = 'model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"Created directory: {model_dir}")

# We need to save both the model and the scaler for the web app to use
# We'll save them together in a dictionary
model_and_scaler = {
    'model': model,
    'scaler': scaler
}

model_path = os.path.join(model_dir, 'fraud_detection_model.pkl')
print(f"Saving the trained model and scaler to {model_path}...")
with open(model_path, 'wb') as file:
    pickle.dump(model_and_scaler, file)

print("Model and scaler saved successfully!")
print("You can now run 'app.py' to start the web application.")

