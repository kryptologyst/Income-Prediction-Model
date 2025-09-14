# main.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

DATA_PATH = 'income_data.csv'
MODEL_PATH = 'income_prediction_model.pkl'
SCALER_PATH = 'scaler.pkl'
ENCODERS_PATH = 'label_encoders.pkl'

def load_data(path):
    """Loads data from a CSV file."""
    return pd.read_csv(path)

def preprocess_data(df):
    """Encodes categorical variables and scales features."""
    le_edu = LabelEncoder()
    le_occ = LabelEncoder()
    
    df['Education'] = le_edu.fit_transform(df['Education'])
    df['Occupation'] = le_occ.fit_transform(df['Occupation'])
    
    label_encoders = {'Education': le_edu, 'Occupation': le_occ}
    
    X = df.drop('Income', axis=1)
    y = df['Income']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, label_encoders

def train_model(X_train, y_train):
    """Trains a linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and prints metrics."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    return y_pred

def visualize_results(y_test, y_pred):
    """Visualizes the actual vs. predicted values."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    plt.title('Actual vs. Predicted Income')
    plt.xlabel('Actual Income')
    plt.ylabel('Predicted Income')
    plt.grid(True)
    plt.show()

def save_artifacts(model, scaler, encoders):
    """Saves the model, scaler, and encoders."""
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    print(f"Model and artifacts saved to {MODEL_PATH}, {SCALER_PATH}, and {ENCODERS_PATH}")

if __name__ == "__main__":
    # Load and preprocess data
    df = load_data(DATA_PATH)
    X_scaled, y, scaler, label_encoders = preprocess_data(df.copy()) # Use copy to avoid modifying original df
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model on the test set:")
    y_pred = evaluate_model(model, X_test, y_test)
    
    # Visualize results
    visualize_results(y_test, y_pred)
    
    # Save model and artifacts
    save_artifacts(model, scaler, label_encoders)
