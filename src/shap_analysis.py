import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use('Agg')  # Headless mode for saving plots without GUI
import matplotlib.pyplot as plt
import shap

def run_shap_analysis():
    # Define paths
    project_dir = "d:/Earthquake_Project"
    model_dir = os.path.join(project_dir, "models")
    labeled_path = os.path.join(project_dir, "labeled_seismic_features.csv")
    
    # Check that required files exist
    assert os.path.exists(labeled_path), f"Dataset not found at {labeled_path}"
    assert os.path.exists(os.path.join(model_dir, "random_forest_model.joblib")), "RF model not found"
    assert os.path.exists(os.path.join(model_dir, "feature_names.joblib")), "Feature names not found"
    
    # Load model and feature names
    print("Loading models and feature names...")
    rf = joblib.load(os.path.join(model_dir, "random_forest_model.joblib"))
    feature_cols = joblib.load(os.path.join(model_dir, "feature_names.joblib"))
    
    # Load dataset
    print("Loading engineered dataset...")
    df = pd.read_csv(labeled_path)
    
    # Test set: years 2019-2024
    test_df = df[df['year'] >= 2019].copy()
    X_test = test_df[feature_cols]
    y_test = test_df['risk_label']
    
    # Predict to find true positives/negatives for local explanations
    y_pred = rf.predict(X_test)
    test_df['y_pred'] = y_pred
    
    # Initialize TreeExplainer
    print("Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(rf)
    
    # Select sample from test set for SHAP computation
    # Using 300 samples for a balance of robust statistics and fast computation
    sample_size = min(300, len(X_test))
    sample_df = X_test.sample(sample_size, random_state=42)
    print(f"Calculating SHAP values for {sample_size} test samples...")
    shap_values = explainer.shap_values(sample_df)
    
    # Handle multi-class outputs which differ by SHAP/sklearn version
    classes = rf.classes_
    print(f"Model classes: {classes}")
    
    # 1. Compute Global SHAP Feature Importance
    print("\n" + "="*50)
    print("--- 1. GLOBAL SHAP FEATURE IMPORTANCE ---")
    print("="*50)
    
    # In SHAP, multi-class shap_values is typically a list of len (n_classes),
    # where each element is an array of shape (n_samples, n_features).
    # We take the mean absolute SHAP value across all samples and classes.
    if isinstance(shap_values, list):
        # List of arrays: mean absolute SHAP value for each class, then average across classes
        mean_abs_shap_by_class = {
            classes[c_idx]: np.abs(shap_values[c_idx]).mean(axis=0)
            for c_idx in range(len(classes))
        }
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        # Array of shape (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
        if len(shap_values.shape) == 3:
            # Check dimensions to determine where classes index is
            if shap_values.shape[0] == len(classes):
                mean_abs_shap_by_class = {
                    classes[c_idx]: np.abs(shap_values[c_idx]).mean(axis=0)
                    for c_idx in range(len(classes))
                }
                mean_abs_shap = np.abs(shap_values).mean(axis=(0, 1))
            else:
                mean_abs_shap_by_class = {
                    classes[c_idx]: np.abs(shap_values[:, :, c_idx]).mean(axis=0)
                    for c_idx in range(len(classes))
                }
                mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
        else:
            # Fallback if binary/1D
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            mean_abs_shap_by_class = {classes[0]: mean_abs_shap}
            
    # Combine into DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'mean_abs_shap': mean_abs_shap
    })
    
    # Add individual class contributions if available
    for cls in classes:
        if cls in mean_abs_shap_by_class:
            importance_df[f'mean_abs_shap_{cls}'] = mean_abs_shap_by_class[cls]
            
    importance_df = importance_df.sort_values(by='mean_abs_shap', ascending=False).reset_index(drop=True)
    
    # Save global SHAP importance to CSV
    shap_out_path = os.path.join(project_dir, "shap_global_importance.csv")
    importance_df.to_csv(shap_out_path, index=False)
    print(f"Global SHAP feature importance saved to: {shap_out_path}")
    print(importance_df.to_string())
    
    # 2. Generate and Save SHAP Summary Plot
    print("\n" + "="*50)
    print("--- 2. GENERATING SHAP SUMMARY PLOT ---")
    print("="*50)
    plt.figure(figsize=(10, 7))
    # SHAP summary plot for multi-class yields a bar plot colored by class
    shap.summary_plot(shap_values, sample_df, feature_names=feature_cols, show=False)
    plt.title("SHAP Multi-Class Feature Importance (Test Set Sample)", fontsize=14, pad=15)
    plt.tight_layout()
    plot_path = os.path.join(project_dir, "shap_summary_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"SHAP summary plot saved to: {plot_path}")
    
    # 3. Generate Local Explanations
    print("\n" + "="*50)
    print("--- 3. LOCAL SHAP EXPLANATIONS (TRUE POSITIVES) ---")
    print("="*50)
    
    # Find one clear True Positive example for High, Medium, and Low risk
    tp_high = test_df[(test_df['risk_label'] == 'High') & (test_df['y_pred'] == 'High')].head(1)
    tp_med = test_df[(test_df['risk_label'] == 'Medium') & (test_df['y_pred'] == 'Medium')].head(1)
    tp_low = test_df[(test_df['risk_label'] == 'Low') & (test_df['y_pred'] == 'Low')].head(1)
    
    examples = {'High': tp_high, 'Medium': tp_med, 'Low': tp_low}
    
    for name, exp_df in examples.items():
        if len(exp_df) > 0:
            idx = exp_df.index[0]
            x_inst = X_test.loc[[idx]]
            
            # Compute SHAP values for this specific instance
            sv_inst = explainer.shap_values(x_inst)
            class_idx = list(classes).index(name)
            
            # The contributions for the predicted class
            if isinstance(sv_inst, list):
                contributions = sv_inst[class_idx][0]
            else:
                if len(sv_inst.shape) == 3:
                    if sv_inst.shape[0] == len(classes):
                        contributions = sv_inst[class_idx, 0, :]
                    else:
                        contributions = sv_inst[0, :, class_idx]
                else:
                    contributions = sv_inst[0]
            
            inst_contrib = pd.DataFrame({
                'feature': feature_cols,
                'feature_value': x_inst.iloc[0].values,
                'shap_value': contributions
            })
            
            # Add column explaining direction of impact
            inst_contrib['direction'] = inst_contrib['shap_value'].apply(lambda x: 'Increases Risk' if x > 0 else 'Decreases Risk')
            inst_contrib['abs_shap'] = inst_contrib['shap_value'].abs()
            inst_contrib = inst_contrib.sort_values(by='abs_shap', ascending=False).drop(columns=['abs_shap'])
            
            print(f"\nLocal Explanation for True Positive {name} Risk Prediction:")
            print(f"Location: Lat {exp_df.iloc[0]['grid_lat']}N, Lon {exp_df.iloc[0]['grid_lon']}E | Year: {exp_df.iloc[0]['year']}")
            print(f"Target HRS: {exp_df.iloc[0]['target_hrs']:.4f}")
            print(inst_contrib.to_string(index=False))
        else:
            print(f"\nNo True Positive example found for {name} Risk")

    # 4. Compare SHAP Importance vs Random Forest Gini Importance
    print("\n" + "="*50)
    print("--- 4. SHAP IMPORTANCE VS. RANDOM FOREST IMPORTANCE ---")
    print("="*50)
    rf_importances = rf.feature_importances_
    
    comparison_df = pd.DataFrame({
        'feature': feature_cols,
        'rf_gini_importance': rf_importances
    })
    
    # Merge with global SHAP importance
    comparison_df = comparison_df.merge(importance_df[['feature', 'mean_abs_shap']], on='feature')
    
    # Add rankings
    comparison_df['rf_rank'] = comparison_df['rf_gini_importance'].rank(ascending=False).astype(int)
    comparison_df['shap_rank'] = comparison_df['mean_abs_shap'].rank(ascending=False).astype(int)
    comparison_df = comparison_df.sort_values(by='shap_rank').reset_index(drop=True)
    
    # Save comparison to CSV
    comp_path = os.path.join(project_dir, "shap_vs_rf_importance.csv")
    comparison_df.to_csv(comp_path, index=False)
    print(f"Comparison saved to: {comp_path}")
    print(comparison_df.to_string())

if __name__ == "__main__":
    run_shap_analysis()
