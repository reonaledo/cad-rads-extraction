import pandas as pd
import numpy as np
import os
from glob import glob
import re
from tqdm import tqdm
from preprocess import evaluate_performance

def get_user_inputs():
    """Get experiment parameters from user"""
    print("\nAvailable options:")
    print("APIs: gemini, claude, chatgpt")
    print("Datasets: InterTest, External")
    
    api = input("\nEnter the API name: ").lower()
    dataset = input("Enter the dataset name (InterTest/External): ")
    n_shot = int(input("Enter the N_SHOT value to analyze: "))
    
    # Validate inputs
    if api not in ['gemini', 'claude', 'chatgpt']:
        raise ValueError("Invalid API name")
    if dataset not in ['InterTest', 'External']:
        raise ValueError("Invalid dataset name")
    if n_shot <= 0:
        raise ValueError("N_SHOT must be positive")
    
    return api, dataset, n_shot

def load_and_filter_data(data_path, dataset):
    """Load data and filter out institution A if external dataset"""
    data = pd.read_csv(data_path)
    data.fillna(value='None', inplace=True)
    
    if dataset == 'External':
        # Load institution information
        institution_data = pd.read_csv('external_w_institution.csv')
        
        # Filter out institution A
        valid_indices = institution_data['Institution'] != 'A (n=30)'
        data = data[valid_indices]
        
        print(f"\nFiltered out Institution A samples.")
        print(f"Remaining samples: {len(data)}")
    
    return data

def extract_experiment_info(filename):
    """Extract experiment information from filename"""
    shot_match = re.search(r'Shot(\d+)_Seed(\d+)', filename)
    if shot_match:
        n_shot = int(shot_match.group(1))
        seed = int(shot_match.group(2))
        return n_shot, seed
    return None, None

def evaluate_single_result(result_file, labels, dataset):
    """Evaluate performance for a single result file"""
    try:
        # Read predictions
        pred_df = pd.read_excel(result_file)
        
        if dataset == 'External':
            # Load institution information
            institution_data = pd.read_csv('external_w_institution.csv')
            
            # Filter out institution A
            valid_indices = institution_data['Institution'] != 'A (n=30)'
            pred_df = pred_df[valid_indices]
        
        # Extract predictions
        pred = pred_df.iloc[:, -9:-1]
        
        # Clean up column names
        pred.columns = pred.columns.str.replace('.1', '', regex=True)
        
        # Ensure columns match labels
        pred = pred[labels.iloc[:,:-1].columns]
        
        # Fix CAD-RADS 4/4A encoding
        pred.loc[pred['CAD-RADS']=='4', 'CAD-RADS'] = '4A'
        
        # Calculate performance metrics
        metrics = evaluate_performance(labels, pred)
        
        # Extract experiment information
        n_shot, seed = extract_experiment_info(result_file)
        
        # Create result dictionary
        result = {
            'n_shot': n_shot,
            'seed': seed,
            'filename': os.path.basename(result_file)
        }
        
        # Flatten metrics dictionary and only include Accuracy and F1 Score
        for key, value in metrics.items():
            if isinstance(value, dict):
                if 'Accuracy' in value:
                    result[f"{key}_Accuracy"] = value['Accuracy']
                if 'F1 Score' in value:
                    result[f"{key}_F1"] = value['F1 Score']
                
        return result
        
    except Exception as e:
        print(f"Error processing {result_file}: {str(e)}")
        return None

def main():
    try:
        # Get user inputs
        API, DATASET, TARGET_N_SHOT = get_user_inputs()
    except ValueError as e:
        print(f"Error: {str(e)}")
        return
    
    # Configuration
    VERSION = 'v1.1' if DATASET == 'InterTest' else 'v6.2'
    RESULTS_DIR = 'results' if DATASET == 'InterTest' else 'External_repeated/results'
    
    # Load ground truth data
    data_path = f'sample_processed_internal_test_50_{VERSION}.csv' if DATASET == 'InterTest' else f'sample_processed_v6.3(외부병원).csv'
    data = load_and_filter_data(data_path, DATASET)
    labels = data.iloc[:,4:]
    
    # Get all result files matching the criteria
    pattern = f'result_1028_{API}_{DATASET}_{VERSION}_CoT_Shot{TARGET_N_SHOT}_Seed*.xlsx'
    result_files = glob(os.path.join(RESULTS_DIR, pattern))
    
    if not result_files:
        print(f"\nNo result files found matching the criteria:")
        print(f"API: {API}")
        print(f"Dataset: {DATASET}")
        print(f"N_SHOT: {TARGET_N_SHOT}")
        return
    
    print(f"\nFound {len(result_files)} result files matching the criteria:")
    print(f"API: {API}")
    print(f"Dataset: {DATASET}")
    print(f"N_SHOT: {TARGET_N_SHOT}")
    
    # Evaluate each result file
    all_results = []
    for file in tqdm(result_files, desc=f"Evaluating results"):
        metrics = evaluate_single_result(file, labels, DATASET)
        if metrics:
            all_results.append(metrics)
    
    if not all_results:
        print("No valid results found!")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Get metric columns (excluding n_shot, seed, and filename)
    metric_columns = [col for col in results_df.columns 
                     if col not in ['n_shot', 'seed', 'filename']]
    
    print("\nAvailable metrics in results:")
    print(metric_columns)
    
    # Add timestamp to output filename
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f'performance_summary_{API}_{DATASET}_shot{TARGET_N_SHOT}_noInstA_{timestamp}.xlsx'
    
    # Calculate summary statistics
    summary_stats = {
        'API': API,
        'Dataset': DATASET,
        'N_SHOT': TARGET_N_SHOT,
        'Number of Experiments': len(results_df),
        'Seeds Used': sorted(results_df['seed'].unique())
    }
    
    # Calculate mean and std for each metric
    metric_stats = {}
    for metric in metric_columns:
        metric_stats[f"{metric}_mean"] = results_df[metric].mean()
        metric_stats[f"{metric}_std"] = results_df[metric].std()
    
    summary_stats.update(metric_stats)
    summary_df = pd.DataFrame([summary_stats])
    
    # Find best configurations
    best_configs = []
    for metric in metric_columns:
        best_idx = results_df[metric].idxmax()
        best_configs.append({
            'Metric': metric,
            'Best Score': round(results_df.loc[best_idx, metric], 4),
            'SEED': results_df.loc[best_idx, 'seed']
        })
    
    best_configs_df = pd.DataFrame(best_configs)
    
    # Save results
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        # Detailed results
        results_df.to_excel(writer, sheet_name='Detailed Results', index=False)
        
        # Summary statistics
        summary_df.to_excel(writer, sheet_name='Summary Statistics', index=False)
        
        # Best configurations
        best_configs_df.to_excel(writer, sheet_name='Best Configurations', index=False)
        
        # Create separate summaries for Accuracy and F1 scores
        for metric_type in ['Accuracy', 'F1']:
            metric_cols = [col for col in metric_columns if metric_type in col]
            if metric_cols:  # Only create sheet if metrics exist
                summary_data = results_df[metric_cols].agg(['mean', 'std', 'min', 'max'])
                summary_data.to_excel(writer, sheet_name=f'{metric_type} Summary')
    
    print(f"\nResults saved to {output_filename}")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"API: {API}")
    print(f"Dataset: {DATASET}")
    print(f"N_SHOT: {TARGET_N_SHOT}")
    print(f"Number of experiments: {len(results_df)}")
    print(f"Seeds used: {sorted(results_df['seed'].unique())}")
    
    print("\nPerformance Metrics:")
    for metric in metric_columns:
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
    
    print("\nBest Configurations:")
    for _, row in best_configs_df.iterrows():
        print(f"{row['Metric']}: {row['Best Score']} (SEED={row['SEED']})")

if __name__ == "__main__":
    main()