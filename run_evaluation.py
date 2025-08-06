# run_evaluation.py

# --- Core Python & Evaluation Libraries ---
import os
import pandas as pd
import numpy as np
import joblib
import argparse

# --- Scikit-learn for metrics ---
from sklearn.metrics import accuracy_score, r2_score, mean_squared_log_error
from sklearn.pipeline import Pipeline

# --- Custom Definitions ---
# This line is CRITICAL. It imports all approved custom classes.
from custom_definitions import *

# --- Main Competition Configuration ---
ALL_DAYS_INFO = {
    1: {'task': 'classification', 'metric': 'Accuracy'},
    2: {'task': 'regression', 'metric': 'R2-Score'},
    3: {'task': 'classification', 'metric': 'Accuracy'},
    4: {'task': 'regression', 'metric': 'RMSLE'} # Using RMSLE for Day 4
}

# --- !!! LEAKAGE CONFIGURATION !!! ---
# Define any known leaky columns for specific days if needed.
LEAKY_COLUMNS_DAY_2 = [] # e.g., ['compressive_strength_duplicate_1']


def get_rank_points(rank):
    """Assigns points based on rank."""
    if rank == 1: return 100
    if rank == 2: return 90
    if rank == 3: return 80
    if rank == 4: return 75
    if rank == 5: return 70
    if 6 <= rank <= 10: return 60
    return 25

def get_used_features(pipeline):
    """Inspects a pipeline to find the input features it uses."""
    if not isinstance(pipeline, Pipeline): return []
    if 'preprocessor' in pipeline.named_steps and hasattr(pipeline.named_steps['preprocessor'], 'transformers_'):
        all_features = []
        for name, transformer, features in pipeline.named_steps['preprocessor'].transformers_:
            if name != 'remainder': all_features.extend(features)
        return all_features
    return []

def rmsle(y_true, y_pred):
    """Calculates Root Mean Squared Log Error."""
    y_pred = np.maximum(y_pred, 0) # Ensure no negative predictions
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def validate_day(day_num, task_type):
    """Validates all submissions for a specific day."""
    print(f"--- Starting Validation for Day {day_num} ({task_type}) ---")
    models_dir = f"day{day_num}_submissions"
    validation_dir = f"day{day_num}_validation"
    output_scores_path = f"day{day_num}_scores.csv"

    if not os.path.exists(models_dir) or not os.path.exists(validation_dir):
        print(f"Error: Required directories for Day {day_num} not found. Aborting.")
        return

    try:
        X_val = pd.read_csv(os.path.join(validation_dir, 'X_val.csv'))
        y_val = pd.read_csv(os.path.join(validation_dir, 'y_val.csv')).squeeze()
    except FileNotFoundError as e:
        print(f"Error loading validation data: {e}. Aborting.")
        return

    daily_results = []
    for file_name in os.listdir(models_dir):
        if file_name.endswith(".pkl"):
            participant_name = os.path.splitext(file_name)[0]
            model_path = os.path.join(models_dir, file_name)
            score = 9999.0 if ALL_DAYS_INFO[day_num]['metric'] == 'RMSLE' else -999.0
            model_name = "Load Failed"
            used_leaky_feature = True

            try:
                pipeline = joblib.load(model_path)
                
                used_leaky_feature = False
                if day_num == 2:
                    features_in_pipeline = get_used_features(pipeline)
                    if any(col in LEAKY_COLUMNS_DAY_2 for col in features_in_pipeline):
                        used_leaky_feature = True

                try:
                    if isinstance(pipeline, Pipeline):
                        final_model_object = pipeline.steps[-1][1]
                        model_name = type(final_model_object).__name__
                    else: model_name = "Custom Wrapper"
                except Exception: model_name = "Inspect Error"

                predictions = pipeline.predict(X_val)

                metric_name = ALL_DAYS_INFO[day_num]['metric']
                if metric_name == 'Accuracy':
                    score = accuracy_score(y_val, predictions)
                elif metric_name == 'RMSLE':
                    score = rmsle(y_val, predictions)
                else: # Default to R2-Score
                    score = r2_score(y_val, predictions)
                
                print(f"  [SUCCESS] Evaluated: {participant_name:<25} | Model: {model_name:<20} | Score: {score:.4f}")

            except Exception as e:
                print(f"  [ FAILED] Evaluated: {participant_name:<25} | Reason: {e}")
            
            result_entry = {"Participant": participant_name, ALL_DAYS_INFO[day_num]['metric']: score, "Model": model_name}
            if day_num == 2: result_entry['Used_Leaky'] = used_leaky_feature
            
            daily_results.append(result_entry)

    if daily_results:
        pd.DataFrame(daily_results).to_csv(output_scores_path, index=False)
        print(f"\nDay {day_num} scores saved to {output_scores_path}")

def update_leaderboard():
    """Recalculates the entire leaderboard."""
    print("\n--- Updating Main Leaderboard ---")
    master_leaderboard = pd.DataFrame()

    for day_num, info in ALL_DAYS_INFO.items():
        scores_file = f"day{day_num}_scores.csv"
        if os.path.exists(scores_file):
            print(f"Processing scores from {scores_file}...")
            daily_df = pd.read_csv(scores_file)
            metric_col = info['metric']
            
            if 'Model' not in daily_df.columns: daily_df['Model'] = 'Legacy'
            
            # --- UPDATED RANKING LOGIC ---
            # For error metrics (like RMSLE), lower is better (ascending=True).
            # For score metrics (like R2), higher is better (ascending=False).
            is_error_metric = metric_col == 'RMSLE'
            
            if f'Used_Leaky' in daily_df.columns:
                # Sort by 'Used_Leaky' first (False comes before True), then by score
                daily_df.sort_values(by=['Used_Leaky', metric_col], ascending=[True, is_error_metric], inplace=True)
                daily_df['Rank'] = range(1, len(daily_df) + 1)
            else:
                daily_df['Rank'] = daily_df[metric_col].rank(method='min', ascending=is_error_metric)
            
            daily_df[f'Day_{day_num}_Points'] = daily_df['Rank'].apply(get_rank_points)
            
            rename_dict = {metric_col: f'Day_{day_num}_Score', 'Model': f'Day_{day_num}_Model'}
            if 'Used_Leaky' in daily_df.columns:
                rename_dict['Used_Leaky'] = f'Day_{day_num}_Used_Leaky'
            daily_df.rename(columns=rename_dict, inplace=True)
            
            cols_to_merge = ['Participant', f'Day_{day_num}_Score', f'Day_{day_num}_Points', f'Day_{day_num}_Model']
            if f'Day_{day_num}_Used_Leaky' in daily_df.columns:
                cols_to_merge.append(f'Day_{day_num}_Used_Leaky')
            
            daily_summary = daily_df[cols_to_merge]

            if master_leaderboard.empty:
                master_leaderboard = daily_summary
            else:
                master_leaderboard = pd.merge(master_leaderboard, daily_summary, on='Participant', how='outer')

    if master_leaderboard.empty:
        print("No score files found.")
        return

    score_cols = [c for c in master_leaderboard.columns if '_Score' in c]
    point_cols = [c for c in master_leaderboard.columns if '_Points' in c]
    model_cols = [c for c in master_leaderboard.columns if '_Model' in c]
    leaky_cols = [c for c in master_leaderboard.columns if '_Used_Leaky' in c]
    
    master_leaderboard.loc[:, score_cols] = master_leaderboard.loc[:, score_cols].fillna(0)
    master_leaderboard.loc[:, point_cols] = master_leaderboard.loc[:, point_cols].fillna(0).astype(int)
    master_leaderboard.loc[:, model_cols] = master_leaderboard.loc[:, model_cols].fillna("N/A")
    master_leaderboard.loc[:, leaky_cols] = master_leaderboard.loc[:, leaky_cols].fillna(True)
    
    master_leaderboard['Total_Points'] = master_leaderboard[point_cols].sum(axis=1)
    master_leaderboard = master_leaderboard.sort_values(by='Total_Points', ascending=False).reset_index(drop=True)
    master_leaderboard['Overall_Rank'] = master_leaderboard.index + 1

    final_cols = ['Overall_Rank', 'Participant', 'Total_Points']
    for day in sorted(ALL_DAYS_INFO.keys()):
        base = [f"Day_{day}_Score", f"Day_{day}_Points", f"Day_{day}_Model"]
        if f"Day_{day}_Used_Leaky" in master_leaderboard.columns:
            base.append(f"Day_{day}_Used_Leaky")
        for col in base:
            if col in master_leaderboard.columns: final_cols.append(col)
            
    master_leaderboard = master_leaderboard[final_cols]
    master_leaderboard.to_csv('leaderboard.csv', index=False)
    print("\nLeaderboard successfully updated.")
    print("--- CURRENT LEADERBOARD ---")
    print(master_leaderboard.to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated evaluation script for the AIRAC Challenge.")
    parser.add_argument("--day", type=int, required=True, help="The challenge day number to validate.")
    args = parser.parse_args()

    if args.day not in ALL_DAYS_INFO:
        print(f"Error: Day {args.day} is not valid.")
    else:
        validate_day(args.day, ALL_DAYS_INFO[args.day]['task'])
        update_leaderboard()
    
