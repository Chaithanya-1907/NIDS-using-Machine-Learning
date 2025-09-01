# src/train.py
"""
This script handles the complete model training process, including hyperparameter tuning.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time
import os

from config import RAW_DATA_PATH, TEST_DATA_PATH, MODEL_PATH, COLUMNS, TARGET_COLUMN, CATEGORICAL_FEATURES

def train_nids():
    """Orchestrates the training, evaluation, and saving of the NIDS model."""
    print("--- NIDS Model Training Script Started ---")

    # Load Data
    print(f"Loading data...")
    train_df = pd.read_csv(RAW_DATA_PATH, header=None, names=COLUMNS)
    test_df = pd.read_csv(TEST_DATA_PATH, header=None, names=COLUMNS)

    # Preprocess Data
    train_df[TARGET_COLUMN] = train_df[TARGET_COLUMN].apply(lambda x: 0 if x == 'normal' else 1)
    test_df[TARGET_COLUMN] = test_df[TARGET_COLUMN].apply(lambda x: 0 if x == 'normal' else 1)

    X_train = train_df.drop([TARGET_COLUMN, 'difficulty'], axis=1)
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop([TARGET_COLUMN, 'difficulty'], axis=1)
    y_test = test_df[TARGET_COLUMN]

    numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()

    # Build Preprocessing and Model Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LinearSVC(random_state=42, dual=False, max_iter=2000))
    ])

    # --- NEW: Hyperparameter Tuning using GridSearchCV ---
    print("\nStarting hyperparameter tuning with GridSearchCV...")
    # WARNING: This can be slow. For a quick test, use a smaller list of values.
    # The 'classifier__' prefix is used to target a parameter within the pipeline.
    param_grid = {
        'classifier__C': [0.1, 1, 10]  # C is the regularization parameter for SVM
    }

    # cv=3 means 3-fold cross-validation. n_jobs=-1 uses all available CPU cores.
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=2)

    # Fit GridSearchCV on the training data. This will find the best model.
    grid_search.fit(X_train, y_train)

    print(f"\nBest parameters found: {grid_search.best_params_}")

    # The best model is stored in grid_search.best_estimator_
    best_model = grid_search.best_estimator_
    # --- End of New Section ---

    # Evaluate the BEST model on the test set
    print("\n--- Model Evaluation (using best model from tuning) ---")
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Normal (0)', 'Attack (1)'])

    print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(report)

    # Save the BEST trained model pipeline
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"\nBest model saved successfully to: {MODEL_PATH}")
    print("--- Script Finished ---")

if __name__ == '__main__':
    train_nids()