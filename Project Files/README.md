# Flood Prediction using Machine Learning

## Overview
This project is a Machine Learning-based web application that predicts the risk of flooding based on historical meteorological data. It takes in various parameters such as rainfall across different seasons and cloud cover, and uses a trained XGBoost classifier to provide a binary prediction (Flood Risk / No Flood Risk) along with a confidence percentage.

## Project Structure

```text
📦 internship
├── 📁 Dataset
│   └── 📄 flood dataset.xlsx        # Raw meteorological dataset
├── 📁 Flask
│   ├── 📄 app.py                    # Flask web server and backend API
│   ├── 📄 floods.save               # Serialized XGBoost model for predictions
│   ├── 📄 transform.save            # Serialized StandardScaler
│   └── 📁 templates
│       └── 📄 index.html            # Web interface frontend
├── 📁 Training
│   └── 📄 floods.ipynb              # EDA and experimental notebook
└── 📄 train_model.py                # Automated model training pipeline
```

## Features
- **Predictive Analytics**: High-accuracy flood risk prediction using an XGBoost Classifier.
- **Data Normalization**: Uses `StandardScaler` to ensure input values align with model training distributions.
- **Web Interface**: User-friendly HTML frontend to input weather parameters easily.
- **Microservices Architecture**: Model training pipeline is completely separate from the inference web server.

## Installation & Setup

### Prerequisites
Ensure you have Python 3.8 or higher installed. You will also need the following dependencies:
```bash
pip install numpy pandas scikit-learn xgboost joblib flask openpyxl
```

### 1. Model Training (Optional)
If you want to retrain the model based on new data from the `Dataset` folder, run the training pipeline:
```bash
python train_model.py
```
*Note: This will overwrite the existing `floods.save` and `transform.save` in the `Flask/` directory with newly trained versions.*

### 2. Running the Web Application
Navigate into the Flask folder and execute the backend server:
```bash
cd Flask
python app.py
```

### 3. Usage
Once the server is running, open your favorite web browser and navigate to:
```
http://127.0.0.1:5000/
```
Enter the required features (Cloud Cover, Annual Rainfall, Jan-Feb Rainfall, etc.) and hit predict to see the flood risk assessment.


