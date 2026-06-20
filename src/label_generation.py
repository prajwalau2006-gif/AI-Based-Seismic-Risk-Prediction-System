import pandas as pd
import numpy as np
import os

def generate_labels():
    input_path = "d:/Earthquake_Project/engineered_seismic_features.csv"
    output_path = "d:/Earthquake_Project/labeled_seismic_features.csv"
    
    print("Loading engineered features...")
    df = pd.read_csv(input_path)
    
    # 1. Compute target Heuristic Risk Score (HRS)
    # HRS = log10(target_count + 1) + 2.0 * target_max_mag
    # If target_count == 0, risk score is 0.0
    print("Calculating Heuristic Risk Scores...")
    df['target_hrs'] = np.where(
        df['target_count'] > 0,
        np.log10(df['target_count'] + 1) + 2.0 * df['target_max_mag'],
        0.0
    )
    
    # 2. Determine thresholds using percentiles to manage class imbalance
    p50 = np.percentile(df['target_hrs'], 50)
    p85 = np.percentile(df['target_hrs'], 85)
    
    print("\n--- RISK THRESHOLDS ---")
    print(f"Low -> Medium Threshold (50th percentile): {p50:.4f}")
    print(f"Medium -> High Threshold (85th percentile): {p85:.4f}")
    
    # 3. Classify scores into Low, Medium, High Risk labels
    def get_risk_label(hrs):
        if hrs <= p50:
            return 'Low'
        elif hrs <= p85:
            return 'Medium'
        else:
            return 'High'
            
    df['risk_label'] = df['target_hrs'].apply(get_risk_label)
    
    # 4. Save the labeled dataset
    df.to_csv(output_path, index=False)
    print(f"\nLabeled dataset saved to {output_path}")
    
    # 5. Output Verification Statistics
    print("\n--- CLASS DISTRIBUTION ---")
    counts = df['risk_label'].value_counts()
    percentages = df['risk_label'].value_counts(normalize=True) * 100
    for label in ['Low', 'Medium', 'High']:
         print(f"{label} Risk: {counts[label]} rows ({percentages[label]:.2f}%)")
         
    print("\n--- GEOPHYSICAL CLASS PROFILE ---")
    profile = df.groupby('risk_label')[['target_count', 'target_max_mag']].agg(['min', 'mean', 'max', 'count'])
    print(profile)

if __name__ == "__main__":
    generate_labels()
