# Income Prediction Model

This project builds a simple linear regression model to predict a person's annual income based on their age, education, occupation, and hours worked per week. The project includes a command-line script for training the model and a Streamlit web application for interactive predictions.

## Project Structure

- `main.py`: The main script for training the model. It loads data, preprocesses it, trains a linear regression model, evaluates it, and saves the trained model and other necessary artifacts.
- `app.py`: A Streamlit web application that loads the trained model and allows users to get income predictions by entering their information.
- `income_data.csv`: A small, mock dataset used for training the model.
- `requirements.txt`: A list of all the Python libraries required to run the project.
- `income_prediction_model.pkl`: The saved file for the trained linear regression model.
- `scaler.pkl`: The saved StandardScaler object used for feature scaling.
- `label_encoders.pkl`: A dictionary of saved LabelEncoder objects for categorical features.
- `.gitignore`: Specifies which files and directories to exclude from version control.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv env
    source env/bin/activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  **Train the model:**

    To train the model and generate the necessary `.pkl` files, run the `main.py` script:

    ```bash
    python3 main.py
    ```

    This will train the model, print evaluation metrics, and save the model, scaler, and encoders.

2.  **Run the web application:**

    To start the Streamlit web app, run the `app.py` script:

    ```bash
    streamlit run app.py
    ```

    You can then access the application in your web browser at `http://localhost:8501`.
