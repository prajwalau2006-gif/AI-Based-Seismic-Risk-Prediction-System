import pandas as pd
import numpy as np
import os

def engineer_features():
    cleaned_path = "d:/Earthquake_Project/cleaned_japan_earthquake_dataset.csv"
    output_path = "d:/Earthquake_Project/engineered_seismic_features.csv"
    
    print("Loading cleaned dataset...")
    df = pd.read_csv(cleaned_path)
    df['time'] = pd.to_datetime(df['time'], format='mixed')
    
    # 1. Spatial Discretization (1.0 degree lat/lon grids)
    df['grid_lat'] = df['latitude'].round(0).astype(int)
    df['grid_lon'] = df['longitude'].round(0).astype(int)
    
    # Identify unique regions and years
    unique_regions = df.groupby(['grid_lat', 'grid_lon']).size().reset_index()[['grid_lat', 'grid_lon']]
    years = sorted(df['year'].unique())
    print(f"Total unique spatial regions (1x1 degree cells): {len(unique_regions)}")
    
    # Prepare panel records
    panel_records = []
    
    for idx, row in unique_regions.iterrows():
        lat, lon = row['grid_lat'], row['grid_lon']
        region_df = df[(df['grid_lat'] == lat) & (df['grid_lon'] == lon)]
        
        # Sliding windows from 2002 to 2024
        # We need historical years Y-2, Y-1, Y to predict target year Y+1
        # E.g. Y=2024 (history: 2022, 2023, 2024; target: 2025)
        for y in range(2002, 2025):
            history_df = region_df[(region_df['year'] >= y-2) & (region_df['year'] <= y)]
            recent_df = region_df[region_df['year'] == y]
            target_df = region_df[region_df['year'] == y+1]
            
            # Frequency
            count_3yr = len(history_df)
            count_1yr = len(recent_df)
            
            # Magnitude stats
            mean_mag_3yr = history_df['mag'].mean() if count_3yr > 0 else 4.0
            max_mag_3yr = history_df['mag'].max() if count_3yr > 0 else 4.0
            mean_mag_1yr = recent_df['mag'].mean() if count_1yr > 0 else 4.0
            
            # Depth stats
            mean_depth_3yr = history_df['depth'].mean() if count_3yr > 0 else 0.0
            mean_depth_1yr = recent_df['depth'].mean() if count_1yr > 0 else 0.0
            
            # Energy release (Proportional to 10^(1.5 * magnitude))
            energy_3yr = np.sum(10**(1.5 * history_df['mag'])) if count_3yr > 0 else 0.0
            energy_1yr = np.sum(10**(1.5 * recent_df['mag'])) if count_1yr > 0 else 0.0
            
            # Trends
            avg_hist_freq = (count_3yr - count_1yr) / 2.0
            freq_trend = count_1yr / (avg_hist_freq + 0.1)
            
            mag_trend = mean_mag_1yr - (history_df[history_df['year'] < y]['mag'].mean() if (count_3yr - count_1yr) > 0 else 4.0)
            
            # Targets (actual activity in the target year Y+1)
            target_count = len(target_df)
            target_max_mag = target_df['mag'].max() if target_count > 0 else 0.0
            target_energy = np.sum(10**(1.5 * target_df['mag'])) if target_count > 0 else 0.0
            
            panel_records.append({
                'grid_lat': lat,
                'grid_lon': lon,
                'year': y,
                'eq_count_3yr': count_3yr,
                'eq_count_1yr': count_1yr,
                'mean_mag_3yr': mean_mag_3yr,
                'max_mag_3yr': max_mag_3yr,
                'mean_depth_3yr': mean_depth_3yr,
                'seismic_energy_3yr': energy_3yr,
                'seismic_energy_1yr': energy_1yr,
                'freq_trend': freq_trend,
                'mag_trend': mag_trend,
                # Target metadata to generate risk categories
                'target_count': target_count,
                'target_max_mag': target_max_mag,
                'target_energy': target_energy
            })
            
    panel_df = pd.DataFrame(panel_records)
    print(f"Engineered dataset compiled. Shape: {panel_df.shape}")
    
    panel_df.to_csv(output_path, index=False)
    print(f"Engineered dataset successfully saved to {output_path}!")

if __name__ == "__main__":
    engineer_features()
