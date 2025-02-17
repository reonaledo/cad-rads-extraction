import pandas as pd
import numpy as np
import os
from glob import glob
import re
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import evaluate_performance

def get_user_inputs():
    """Get experiment parameters from user"""
    print("\nAvailable options:")
    print("APIs: gemini, claude, chatgpt")
    print("Datasets: InterTest, External")
    
    api = input("\nEnter the API name: ").lower()
    dataset = input("Enter the dataset name (InterTest/External): ")
    
    if api not in ['gemini', 'claude', 'chatgpt']:
        raise ValueError("Invalid API name")
    if dataset not in ['InterTest', 'External']:
        raise ValueError("Invalid dataset name")
    
    return api, dataset

def load_and_filter_data(data_path, dataset):
    """Load data and filter to keep only institution A if external dataset"""
    data = pd.read_csv(data_path)
    data.fillna(value='None', inplace=True)
    
    if dataset == 'External':
        # Load institution information
        institution_data = pd.read_csv('external_w_institution.csv')
        
        # Filter to keep only institution A
        valid_indices = institution_data['Institution'] == 'A (n=30)'
        data = data[valid_indices]
        
        print(f"\nKept only Institution A samples.")
        print(f"Number of samples: {len(data)}")
    
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
            
            # Filter to keep only institution A
            valid_indices = institution_data['Institution'] == 'A (n=30)'
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
        
        # Flatten metrics dictionary and only include Accuracy 
        for key, value in metrics.items():
            if isinstance(value, dict) and 'Accuracy' in value:
                result[f"{key}_Acc"] = value['Accuracy']
                
        return result
        
    except Exception as e:
        print(f"Error processing {result_file}: {str(e)}")
        return None

def perform_correlation_analysis(results_df, metric_columns):
    """Perform Spearman's correlation analysis for each metric"""
    correlation_results = []
    
    for metric in metric_columns:
        # Calculate correlation
        correlation, p_value = stats.spearmanr(results_df['n_shot'], results_df[metric])
        
        # Create result dictionary
        result = {
            'Metric': metric,
            'Correlation': correlation,
            'P_value': p_value
        }
        correlation_results.append(result)
        
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=results_df, x='n_shot', y=metric, alpha=0.5)
        plt.title(f'Correlation between Number of Shots and {metric}\n(Institution A Only)')
        plt.xlabel('Number of Shots')
        plt.ylabel('Accuracy')
        
        # Add text with correlation and p-value
        plt.text(0.05, 0.95, 
                f'ρ = {correlation:.3f}\np = {p_value:.3f}', 
                transform=plt.gca().transAxes)
        
        # Save plot
        plt.savefig(f'correlation_plot_{metric}_InstA.png')
        plt.close()
    
    return pd.DataFrame(correlation_results)

def main():
    try:
        # Get user inputs
        API, DATASET = get_user_inputs()
    except ValueError as e:
        print(f"Error: {str(e)}")
        return
    
    # Configuration
    VERSION = 'v1.1' if DATASET == 'InterTest' else 'v6.2'
    RESULTS_DIR = 'results' if DATASET == 'InterTest' else 'External_repeated/results'
    
    # Load ground truth data
    data_path = f'sample_processed_internal_test_50_{VERSION}.csv' if DATASET == 'InterTest' else f'sample_processed_v6.4(외부병원).csv'
    data = load_and_filter_data(data_path, DATASET)
    labels = data.iloc[:,4:]
    
    # Get all result files for all shot numbers
    pattern = f'result_1028_{API}_{DATASET}_{VERSION}_CoT_Shot*_Seed*.xlsx'
    result_files = glob(os.path.join(RESULTS_DIR, pattern))
    
    if not result_files:
        print(f"\nNo result files found matching the criteria:")
        print(f"API: {API}")
        print(f"Dataset: {DATASET}")
        return
    
    print(f"\nFound {len(result_files)} result files")
    
    # Evaluate each result file
    all_results = []
    for file in tqdm(result_files, desc="Evaluating results"):
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
    
    # Perform correlation analysis
    correlation_df = perform_correlation_analysis(results_df, metric_columns)
    
    # Add timestamp to output filename
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f'correlation_analysis_{API}_{DATASET}_InstA_{timestamp}.xlsx'
    
    # Save results
    correlation_df.to_excel(output_filename, index=False)
    
    print(f"\nCorrelation Analysis Results (Institution A Only):")
    for _, row in correlation_df.iterrows():
        print(f"\n{row['Metric']}:")
        print(f"Spearman's ρ: {row['Correlation']:.3f}")
        print(f"p-value: {row['P_value']:.3f}")

if __name__ == "__main__":
    main()