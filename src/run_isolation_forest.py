import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

def run_isolation_forest():
    input_path = "d:/Earthquake_Project/engineered_seismic_features.csv"
    model_dir = "d:/Earthquake_Project/models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Load dataset
    df = pd.read_csv(input_path)
    
    # Extract feature columns
    feature_cols = [
        'eq_count_3yr', 'eq_count_1yr', 
        'mean_mag_3yr', 'max_mag_3yr', 'mean_depth_3yr', 
        'seismic_energy_3yr', 'seismic_energy_1yr', 
        'freq_trend', 'mag_trend'
    ]
    X = df[feature_cols]
    
    # Train Isolation Forest
    contamination_rate = 0.01
    print(f"Training Isolation Forest (contamination={contamination_rate:.2f})...")
    iso = IsolationForest(contamination=contamination_rate, random_state=42)
    df['anomaly_flag'] = iso.fit_predict(X)
    df['anomaly_score'] = iso.decision_function(X)
    
    # Save the model
    model_path = os.path.join(model_dir, "isolation_forest_model.joblib")
    joblib.dump(iso, model_path)
    print(f"Isolation Forest model saved to {model_path}")
    
    # Save anomaly features CSV
    output_path = "d:/Earthquake_Project/seismic_anomalies.csv"
    df.to_csv(output_path, index=False)
    print(f"Anomaly mapping saved to {output_path}")
    
    # Print summary
    anomalies = df[df['anomaly_flag'] == -1]
    print(f"\nSuccessfully identified {len(anomalies)} anomalous region-years out of {len(df)} total rows.")
    
if __name__ == "__main__":
    run_isolation_forest()
