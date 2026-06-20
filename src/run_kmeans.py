import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

def run_kmeans():
    input_path = "d:/Earthquake_Project/engineered_seismic_features.csv"
    model_dir = "d:/Earthquake_Project/models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Load dataset
    df = pd.read_csv(input_path)
    
    # Aggregate data per unique spatial region to identify geographical hotspots
    region_df = df.groupby(['grid_lat', 'grid_lon']).agg({
        'eq_count_1yr': 'mean',
        'max_mag_3yr': 'max',
        'mean_mag_3yr': 'mean',
        'mean_depth_3yr': 'mean',
        'seismic_energy_1yr': 'mean'
    }).reset_index()
    
    region_df.rename(columns={
        'eq_count_1yr': 'avg_annual_count',
        'max_mag_3yr': 'max_magnitude',
        'mean_mag_3yr': 'avg_magnitude',
        'mean_depth_3yr': 'avg_depth',
        'seismic_energy_1yr': 'avg_energy'
    }, inplace=True)
    
    # Prepare features for clustering
    features = ['avg_annual_count', 'max_magnitude', 'avg_magnitude', 'avg_depth', 'avg_energy']
    X = region_df[features]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler
    joblib.dump(scaler, os.path.join(model_dir, "scaler_kmeans.joblib"))
    
    # Set optimal cluster count K=4
    optimal_k = 4
    print(f"Applying K-Means with K={optimal_k}...")
    km = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    region_df['cluster'] = km.fit_predict(X_scaled)
    
    # Save clustering model
    joblib.dump(km, os.path.join(model_dir, "kmeans_model.joblib"))
    
    # Save regional cluster mapping CSV
    output_path = "d:/Earthquake_Project/seismic_regions_clusters.csv"
    region_df.to_csv(output_path, index=False)
    print(f"Cluster mapping saved to {output_path}")
    
    # Print cluster characteristics
    print("\n" + "="*50)
    print(f"--- K-MEANS CLUSTER PROFILES (K = {optimal_k}) ---")
    print("="*50)
    summary = region_df.groupby('cluster').agg({
        'grid_lat': 'count',
        'avg_annual_count': 'mean',
        'max_magnitude': 'mean',
        'avg_magnitude': 'mean',
        'avg_depth': 'mean',
        'avg_energy': 'mean'
    }).rename(columns={'grid_lat': 'cell_count'})
    print(summary)
    
    # Map clusters to risk categories
    # Sort cluster IDs by average annual count to logically assign activity levels
    sorted_clusters = summary.sort_values(by='avg_annual_count').index.tolist()
    
    print("\nLogical Mapping of Clusters to Seismicity Levels:")
    activity_levels = ["Low Activity (Quiet Crustal)", "Low-to-Medium (Deep Mantle Slab)", "Medium-to-High (Active Plate Boundary)", "Catastrophic High (Japan Trench Hotspot)"]
    for idx, c_id in enumerate(sorted_clusters):
         print(f"Cluster {c_id:2d} -> {activity_levels[idx]}")

if __name__ == "__main__":
    run_kmeans()
