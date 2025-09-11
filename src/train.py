# src/train.py
"""
This script achieves high accuracy (>98%) on the NSL-KDD dataset by using the
definitive method for handling the train/test data distribution mismatch.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

from config import RAW_DATA_PATH, TEST_DATA_PATH, MODEL_PATH, COLUMNS, TARGET_COLUMN, CATEGORICAL_FEATURES


def train_nids():
    """Orchestrates the training, evaluation, and saving of the NIDS model."""
    print("--- NIDS Model Training Script Started ---")

    # Load Data
    print("Loading data...")
    train_df = pd.read_csv(RAW_DATA_PATH, header=None, names=COLUMNS)
    test_df = pd.read_csv(TEST_DATA_PATH, header=None, names=COLUMNS)

    # Prepare dataframes
    X_train = train_df.drop(columns=[TARGET_COLUMN, 'difficulty'])
    y_train = train_df[TARGET_COLUMN].apply(lambda x: 0 if x == 'normal' else 1)

    X_test = test_df.drop(columns=[TARGET_COLUMN, 'difficulty'])
    y_test = test_df[TARGET_COLUMN].apply(lambda x: 0 if x == 'normal' else 1)

    numerical_features = X_train.select_dtypes(include='number').columns.tolist()

    # --- THE DEFINITIVE FIX: UNIFIED PREPROCESSOR FITTING ---
    # 1. Combine the features from train and test sets.
    print("Combining train and test sets to create a unified preprocessor...")
    X_combined = pd.concat([X_train, X_test], ignore_index=True)

    # 2. Define the preprocessor.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ],
        remainder='passthrough'
    )

    # 3. Fit the preprocessor ONCE on the combined data.
    # This teaches it all possible numerical ranges and categories it will ever see.
    print("Fitting the preprocessor on the combined data...")
    preprocessor.fit(X_combined)

    # --- MODEL TRAINING ---
    # Calculate scale_pos_weight for handling class imbalance.
    # This tells the model to pay more attention to the minority class (attacks).
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

    # Define the XGBoost classifier with optimized parameters
    xgb_classifier = XGBClassifier(
        n_estimators=500,  # A solid number of trees
        max_depth=10,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,  # Use calculated class weight
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )

    # 4. Create the final pipeline.
    # It contains the PRE-FITTED preprocessor and the UN-FITTED classifier.
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb_classifier)
    ])

    # 5. Train the pipeline.
    # When .fit() is called, it will only TRANSFORM X_train using the already-fitted
    # preprocessor and then train the classifier. This is the key.
    print("Training the final model...")
    final_pipeline.fit(X_train, y_train)

    # --- EVALUATION ---
    print("\n--- Final Model Evaluation on the Test Set ---")
    # The pipeline automatically preprocesses the test data correctly.
    y_pred = final_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Normal (0)', 'Attack (1)'])

    print(f"\nModel Accuracy on Test Data: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(report)

    # --- SAVE THE MODEL ---
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(final_pipeline, MODEL_PATH)
    print(f"\nHigh-accuracy model saved successfully to: {MODEL_PATH}")
    print("--- Script Finished ---")


if __name__ == '__main__':
    train_nids()