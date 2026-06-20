import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import joblib
import os

def train_model():
    input_path = "d:/Earthquake_Project/labeled_seismic_features.csv"
    model_dir = "d:/Earthquake_Project/models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Load dataset
    df = pd.read_csv(input_path)
    
    # 1. Feature selection (Exclude targets to prevent data leakage)
    target_cols = ['target_count', 'target_max_mag', 'target_energy', 'target_hrs', 'risk_label']
    feature_cols = [col for col in df.columns if col not in target_cols]
    
    # 2. Chronological Train/Test Split
    # Train: pivot years 2002-2018 (representing target years 2003-2019)
    # Test: pivot years 2019-2024 (representing target years 2020-2025)
    train_df = df[df['year'] <= 2018]
    test_df = df[df['year'] >= 2019]
    
    X_train = train_df[feature_cols]
    y_train = train_df['risk_label']
    X_test = test_df[feature_cols]
    y_test = test_df['risk_label']
    
    print(f"Training features ({len(feature_cols)}): {feature_cols}")
    print(f"Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")
    
    # 3. Train Random Forest Classifier
    # Using class_weight='balanced' to handle any remaining minor class imbalance
    print("\nTraining Random Forest model...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=12)
    rf.fit(X_train, y_train)
    
    # Save the trained model for later phases
    model_path = os.path.join(model_dir, "random_forest_model.joblib")
    joblib.dump(rf, model_path)
    print(f"Model saved to {model_path}")
    
    # Save feature names list
    feature_list_path = os.path.join(model_dir, "feature_names.joblib")
    joblib.dump(feature_cols, feature_list_path)
    
    # 4. Evaluate on Overall Test Set (2020-2025 target years)
    y_pred = rf.predict(X_test)
    print("\n" + "="*50)
    print("--- OVERALL TEST SET EVALUATION (Target Years 2020-2025) ---")
    print("="*50)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # Explicitly pass labels to ensure correct row/column alignment
    classes = ['Low', 'Medium', 'High']
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=classes))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=classes))
    
    # 5. Evaluate on 2020 Backtesting Set (Pivot Year 2019)
    test_2020 = test_df[test_df['year'] == 2019]
    X_2020 = test_2020[feature_cols]
    y_2020 = test_2020['risk_label']
    y_pred_2020 = rf.predict(X_2020)
    
    print("\n" + "="*50)
    print("--- 2020 BACKTESTING SPECIFIC EVALUATION (Pivot Year 2019) ---")
    print("="*50)
    print(f"2020 Accuracy: {accuracy_score(y_2020, y_pred_2020):.4f}")
    print("\nClassification Report (2020):")
    print(classification_report(y_2020, y_pred_2020, labels=classes))
    
    print("Confusion Matrix (2020):")
    print(confusion_matrix(y_2020, y_pred_2020, labels=classes))
    
    # 6. Feature Importances
    print("\n" + "="*50)
    print("--- FEATURE IMPORTANCES ---")
    print("="*50)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(X_train.shape[1]):
        print(f"{f + 1:2d}. {feature_cols[indices[f]]:20s} : {importances[indices[f]]:.4f}")

if __name__ == "__main__":
    train_model()
