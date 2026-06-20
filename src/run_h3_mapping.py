import pandas as pd
import numpy as np
import joblib
import os
import h3

def run_h3_mapping():
    project_dir = "d:/Earthquake_Project"
    model_dir = os.path.join(project_dir, "models")
    labeled_path = os.path.join(project_dir, "labeled_seismic_features.csv")
    output_path = os.path.join(project_dir, "h3_risk_map.csv")
    
    # Check that required files exist
    assert os.path.exists(labeled_path), f"Dataset not found at {labeled_path}"
    assert os.path.exists(os.path.join(model_dir, "random_forest_model.joblib")), "RF model not found"
    assert os.path.exists(os.path.join(model_dir, "feature_names.joblib")), "Feature names not found"
    
    # Load model and feature names
    print("Loading models and feature names...")
    rf = joblib.load(os.path.join(model_dir, "random_forest_model.joblib"))
    feature_cols = joblib.load(os.path.join(model_dir, "feature_names.joblib"))
    classes = list(rf.classes_)
    
    # Load dataset
    print("Loading labeled dataset...")
    df = pd.read_csv(labeled_path)
    
    # Generate predictions & probabilities for all rows
    print("Generating predictions and probabilities...")
    X = df[feature_cols]
    df['predicted_risk_label'] = rf.predict(X)
    
    probs = rf.predict_proba(X)
    for idx, cls in enumerate(classes):
        df[f'predicted_prob_{cls}'] = probs[:, idx]
        
    # Map each row to H3 Resolution 5 hexagon
    # Center of grid cell is (grid_lat + 0.5, grid_lon + 0.5)
    print("Converting lat-lon grids to H3 Resolution 5 cells...")
    df['h3_cell'] = df.apply(lambda r: h3.latlng_to_cell(r['grid_lat'] + 0.5, r['grid_lon'] + 0.5, 5), axis=1)
    
    # Reorganize columns for clarity
    output_cols = [
        'h3_cell', 'grid_lat', 'grid_lon', 'year',
        'target_hrs', 'risk_label', 'predicted_risk_label'
    ] + [f'predicted_prob_{cls}' for cls in classes] + [
        'eq_count_3yr', 'seismic_energy_3yr', 'mean_depth_3yr'
    ]
    
    h3_df = df[output_cols].rename(columns={
        'target_hrs': 'actual_hrs',
        'risk_label': 'actual_risk_label'
    })
    
    # Save the output CSV
    h3_df.to_csv(output_path, index=False)
    print(f"H3 risk mapping dataset saved successfully to: {output_path}")
    print(f"Total mapped records: {len(h3_df)}")
    
    # Identify top 10 high-risk H3 zones (based on predicted High-risk probability in the test set)
    print("\n" + "="*50)
    print("--- TOP 10 PREDICTED HIGH-RISK H3 ZONES (RECENT YEARS 2019-2024) ---")
    print("="*50)
    
    test_h3 = h3_df[h3_df['year'] >= 2019]
    top_zones = test_h3.groupby('h3_cell').agg({
        'grid_lat': 'first',
        'grid_lon': 'first',
        'predicted_prob_High': 'mean',
        'actual_hrs': 'mean',
        'eq_count_3yr': 'mean',
        'seismic_energy_3yr': 'mean'
    }).sort_values(by='predicted_prob_High', ascending=False).head(10).reset_index()
    
    # Add major regions name based on latitude-longitude coordinates for interpretation
    # 38N, 142E is Tohoku; 35N, 140E is Kanto/Tokyo; 24N, 143E is Izu-Bonin; etc.
    def identify_region(lat, lon):
        if lat >= 37 and lat <= 41 and lon >= 140 and lon <= 144:
            return "Offshore Tohoku (Trench)"
        elif lat >= 34 and lat <= 36 and lon >= 139 and lon <= 141:
            return "Greater Kanto (Tokyo/Chiba)"
        elif lat >= 23 and lat <= 28 and lon >= 141 and lon <= 144:
            return "Izu-Bonin Trench Area"
        elif lat >= 31 and lat <= 34 and lon >= 130 and lon <= 133:
            return "Kyushu / Southern Region"
        elif lat >= 41 and lat <= 46 and lon >= 140 and lon <= 146:
            return "Hokkaido / Kuril Trench"
        elif lat >= 36 and lat <= 39 and lon >= 136 and lon <= 139:
            return "Chubu / Noto Peninsula"
        elif lat >= 24 and lat <= 26 and lon >= 122 and lon <= 126:
            return "Ryukyu Islands / Okinawa"
        else:
            return "Other Seismic Zone"

    top_zones['region_name'] = top_zones.apply(lambda r: identify_region(r['grid_lat'], r['grid_lon']), axis=1)
    
    for idx, row in top_zones.iterrows():
        print(f"{idx+1:2d}. Cell: {row['h3_cell']} | Lat: {row['grid_lat']}N, Lon: {row['grid_lon']}E | Region: {row['region_name']}")
        print(f"    Avg Predicted Prob(High): {row['predicted_prob_High']:.4f} | Avg HRS: {row['actual_hrs']:.4f} | Avg 3Yr Eq Count: {row['eq_count_3yr']:.1f}")
        print(f"    Avg 3Yr Energy: {row['seismic_energy_3yr']:.2e} J\n")

if __name__ == "__main__":
    run_h3_mapping()
