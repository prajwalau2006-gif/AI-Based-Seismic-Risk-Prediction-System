import pandas as pd
import numpy as np
import os
import math
import heapq
import h3

# Haversine distance formula for coordinate distance estimation
def haversine(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)
    
    a = math.sin(d_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    r = 6371.0  # Earth radius in km
    return r * c

class H3RoutePlanner:
    def __init__(self, risk_map_path="d:/Earthquake_Project/h3_risk_map.csv", default_year=2024):
        self.risk_map_path = risk_map_path
        self.default_year = default_year
        self.risk_db = {}
        self.load_risk_map()

    def load_risk_map(self):
        if not os.path.exists(self.risk_map_path):
            print(f"Warning: {self.risk_map_path} not found. Running with default Low Risk everywhere.")
            return
            
        df = pd.read_csv(self.risk_map_path)
        # Use risk profiles for the default year (most recent year)
        df_year = df[df['year'] == self.default_year]
        
        for _, row in df_year.iterrows():
            self.risk_db[row['h3_cell']] = {
                'predicted_risk': row['predicted_risk_label'],
                'actual_risk': row['actual_risk_label'],
                'prob_High': row['predicted_prob_High']
            }
        print(f"Loaded risk profiles for {len(self.risk_db)} H3 cells for year {self.default_year}.")

    def get_cell_risk(self, cell):
        # Default to Low Risk if the cell is not in the database
        if cell in self.risk_db:
            return self.risk_db[cell]['predicted_risk']
        return 'Low'

    def get_cell_cost(self, cell, mode='safest'):
        if mode == 'shortest':
            return 1.0  # Constant step cost
            
        risk = self.get_cell_risk(cell)
        if risk == 'High':
            return 20.0
        elif risk == 'Medium':
            return 5.0
        else:
            return 1.0

    def find_route(self, start_lat, start_lon, end_lat, end_lon, mode='safest'):
        # Convert start/end coordinates to H3 cells at resolution 5
        start_cell = h3.latlng_to_cell(start_lat, start_lon, 5)
        end_cell = h3.latlng_to_cell(end_lat, end_lon, 5)
        
        if start_cell == end_cell:
            return [start_cell], 0.0, 0.0
            
        # Priority queue stores tuples of (f_score, g_score, current_cell, path)
        # f_score = g_score + heuristic
        start_latlng = h3.cell_to_latlng(start_cell)
        end_latlng = h3.cell_to_latlng(end_cell)
        
        h_start = haversine(start_latlng[0], start_latlng[1], end_latlng[0], end_latlng[1])
        open_set = []
        heapq.heappush(open_set, (h_start, 0.0, start_cell, [start_cell]))
        
        # Track visited nodes and their minimum g_score (cost from start)
        g_scores = {start_cell: 0.0}
        
        while open_set:
            f, g, curr, path = heapq.heappop(open_set)
            
            if curr == end_cell:
                # Compute actual physical distance along path
                total_distance = 0.0
                for i in range(len(path) - 1):
                    c1_coords = h3.cell_to_latlng(path[i])
                    c2_coords = h3.cell_to_latlng(path[i+1])
                    total_distance += haversine(c1_coords[0], c1_coords[1], c2_coords[0], c2_coords[1])
                return path, g, total_distance
                
            # Get neighbors (adjacent H3 cells)
            neighbors = set(h3.grid_disk(curr, 1)) - {curr}
            
            for neighbor in neighbors:
                # Validate neighborhood bounds (limit to Japan bounding box coordinates)
                coords = h3.cell_to_latlng(neighbor)
                lat, lon = coords[0], coords[1]
                if not (20.0 <= lat <= 48.0 and 120.0 <= lon <= 150.0):
                    continue  # Skip cells outside Japan bounds
                    
                # Traversal cost: distance-weighted risk cost
                # Since cells are uniform, we multiply distance by risk cost factor
                step_dist = haversine(h3.cell_to_latlng(curr)[0], h3.cell_to_latlng(curr)[1], lat, lon)
                step_cost = step_dist * self.get_cell_cost(neighbor, mode=mode)
                
                tentative_g = g + step_cost
                
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    h_val = haversine(lat, lon, end_latlng[0], end_latlng[1])
                    f_val = tentative_g + h_val
                    heapq.heappush(open_set, (f_val, tentative_g, neighbor, path + [neighbor]))
                    
        return None, float('inf'), 0.0

def run_route_comparison(start_lat, start_lon, end_lat, end_lon, label):
    planner = H3RoutePlanner()
    
    print("\n" + "="*60)
    print(f"--- ROUTE PLANNING: {label} ---")
    print("="*60)
    
    # 1. Shortest Path Route
    path_short, cost_short, dist_short = planner.find_route(start_lat, start_lon, end_lat, end_lon, mode='shortest')
    
    # 2. Safest Path Route
    path_safe, cost_safe, dist_safe = planner.find_route(start_lat, start_lon, end_lat, end_lon, mode='safest')
    
    if not path_short or not path_safe:
        print("Error: Could not find path.")
        return

    print(f"\n[Shortest Distance Mode]")
    print(f"Path Length (Hexagons): {len(path_short)} cells")
    print(f"Accumulated Cost Score: {cost_short:.2f}")
    print(f"Total Physical Distance: {dist_short:.2f} km")
    
    print(f"\n[Safest Risk-Averse Mode]")
    print(f"Path Length (Hexagons): {len(path_safe)} cells")
    print(f"Accumulated Cost Score: {cost_safe:.2f}")
    print(f"Total Physical Distance: {dist_safe:.2f} km")
    
    # Compare risk profiles
    short_risks = [planner.get_cell_risk(c) for c in path_short]
    safe_risks = [planner.get_cell_risk(c) for c in path_safe]
    
    print("\n" + "-"*50)
    print("ROUTE RISK COMPARISON:")
    print("-"*50)
    print(f"Shortest Path Risks: High: {short_risks.count('High')} | Medium: {short_risks.count('Medium')} | Low: {short_risks.count('Low')}")
    print(f"Safest Path Risks:   High: {safe_risks.count('High')} | Medium: {safe_risks.count('Medium')} | Low: {safe_risks.count('Low')}")
    
    # Detail steps for shortest vs safest
    print(f"\nShortest Path steps: {[planner.get_cell_risk(c) for c in path_short]}")
    print(f"Safest Path steps:   {[planner.get_cell_risk(c) for c in path_safe]}")
    
    if short_risks.count('High') > safe_risks.count('High') or short_risks.count('Medium') > safe_risks.count('Medium'):
        print("\n>>> Success: The Safest Route Planner successfully avoided high/medium risk cells!")
    else:
        print("\n>>> Note: The shortest path was already the safest path for this route geometry.")

def run_routing_demo():
    # Scenario 1: Tokyo to Sendai (Inland Valley Path)
    run_route_comparison(35.6762, 139.6503, 38.2682, 140.8694, "TOKYO TO SENDAI (INLAND PATH)")
    
    # Scenario 2: Chiba Coastal Trench to Gunma Inland
    run_route_comparison(34.5, 140.5, 36.5, 139.5, "CHIBA TRENCH TO GUNMA INLAND")

if __name__ == "__main__":
    run_routing_demo()
