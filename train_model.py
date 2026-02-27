"""
Flood Prediction Model Training Script
Replicates the training done in Training/floods.ipynb
and saves the model and scaler to the Flask directory.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from joblib import dump
import xgboost

print("Loading dataset...")
dataset = pd.read_excel('Dataset/flood dataset.xlsx')

print("Dataset shape:", dataset.shape)
print("Columns:", list(dataset.columns))
print("First few rows:")
print(dataset.head(3))

# Features: Cloud Cover, ANNUAL, Jan-Feb, Mar-May, Jun-Sep (columns 2-6)
x = dataset.iloc[:, 2:7].values
# Target: flood (column 10)
y = dataset.iloc[:, 10].values

print("\nFeature columns (2-6):", list(dataset.columns[2:7]))
print("Target column (10):", dataset.columns[10])
print("Feature shape:", x.shape)
print("Target distribution:\n", pd.Series(y).value_counts())

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=10
)

# Standard scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)   # Use transform (not fit_transform) on test set

# Save the scaler to Flask directory
print("\nSaving scaler to Flask/transform.save ...")
dump(sc, 'Flask/transform.save')

# Train classifiers
print("\nTraining classifiers...")
dtree = DecisionTreeClassifier()
Rf = RandomForestClassifier()
knn = KNeighborsClassifier()
xgb = xgboost.XGBClassifier(eval_metric='logloss')

dtree.fit(x_train, y_train)
Rf.fit(x_train, y_train)
knn.fit(x_train, y_train)
xgb.fit(x_train, y_train)

# Evaluate
p1 = dtree.predict(x_test)
p2 = Rf.predict(x_test)
p3 = knn.predict(x_test)
p4 = xgb.predict(x_test)

print("\n=== Model Accuracy ===")
print(f"Decision Tree:  {metrics.accuracy_score(y_test, p1):.4f}")
print(f"Random Forest:  {metrics.accuracy_score(y_test, p2):.4f}")
print(f"KNN:            {metrics.accuracy_score(y_test, p3):.4f}")
print(f"XGBoost:        {metrics.accuracy_score(y_test, p4):.4f}")

print("\nXGBoost Confusion Matrix:")
print(metrics.confusion_matrix(y_test, p4))

# Save the best model (XGBoost) to Flask directory
print("\nSaving XGBoost model to Flask/floods.save ...")
dump(xgb, 'Flask/floods.save')

print("\nDone! Model and scaler saved successfully.")
print("Features used:", list(dataset.columns[2:7]))
