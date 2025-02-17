import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import os
import re
from preprocess import evaluate_performance

def get_unique_classes(result_files, labels):
    """Get unique classes for all categories"""
    unique_classes = {
        'CAD-RADS': set(), 'Plaque Burden': set(), 
        'E': set(), 'N': set(), 'G': set(), 
        'HRP': set(), 'S': set(), 'I': set()
    }
    
    # Add classes from all prediction files
    for file in result_files:
        pred_df = pd.read_excel(file)
        pred = pred_df.iloc[:, -9:-1]
        pred.columns = pred.columns.str.replace('.1', '', regex=True)
        pred = pred[labels.iloc[:,:-1].columns]
        
        # Convert None to '0' for CAD-RADS
        pred.loc[pred['CAD-RADS'].isna(), 'CAD-RADS'] = '0'
        pred.loc[pred['CAD-RADS']=='4', 'CAD-RADS'] = '4A'
        
        for key in unique_classes.keys():
            if key == 'CAD-RADS':
                # For CAD-RADS, convert None to '0' and exclude None from classes
                classes = set(pred[key].replace('None', '0').dropna().unique())
                unique_classes[key].update(classes)
            else:
                unique_classes[key].update(pred[key].dropna().unique())
    
    # Remove 'None' from CAD-RADS classes if exists
    if 'CAD-RADS' in unique_classes:
        unique_classes['CAD-RADS'].discard('None')
    
    # Convert sets to sorted lists
    return {k: sorted(list(v)) for k, v in unique_classes.items()}

def create_confusion_matrix(true_labels, pred_labels, classes):
    """Create confusion matrix with given classes"""
    n = len(classes)
    cm = np.zeros((n, n))
    
    for t, p in zip(true_labels, pred_labels):
        if pd.notna(t) and pd.notna(p):
            i = classes.index(t)
            j = classes.index(p)
            cm[i, j] += 1
    return cm

def load_and_evaluate_results(result_files, labels, unique_classes):
    """Load and evaluate multiple experiment results"""
    all_metrics = []
    all_cms = {key: [] for key in unique_classes.keys()}
    
    for file in result_files:
        # Load and preprocess predictions
        pred_df = pd.read_excel(file)
        pred = pred_df.iloc[:, -9:-1]
        pred.columns = pred.columns.str.replace('.1', '', regex=True)
        pred = pred[labels.iloc[:,:-1].columns]

        pred.loc[pred['CAD-RADS'].isna(), 'CAD-RADS'] = '0'
        pred.loc[pred['CAD-RADS']=='None', 'CAD-RADS'] = '0'
        pred.loc[pred['CAD-RADS']=='4', 'CAD-RADS'] = '4A'
        
        # Get metrics using evaluate_performance
        metrics = evaluate_performance(labels, pred)
        all_metrics.append(metrics)
        
        # Create standardized confusion matrices
        for key in unique_classes.keys():
            if key == 'Plaque Burden':
                mask = (labels['S'] == 0) & (labels['CAC_available'] == 1)
                true_labels = labels[key][mask]
                pred_labels = pred[key][mask]
            else:
                true_labels = labels[key]
                pred_labels = pred[key]
            
            cm = create_confusion_matrix(true_labels, pred_labels, unique_classes[key])
            all_cms[key].append(cm)
    
    return all_metrics, all_cms

def calculate_aggregated_cms(all_metrics, all_cms, unique_classes):
    """Calculate mean and std of confusion matrices"""
    aggregated_cms = {}
    
    for key in unique_classes.keys():
        matrices = all_cms[key]
        if matrices:
            # Convert list of matrices to 3D numpy array
            matrices_array = np.array(matrices)
            
            # Calculate mean and std
            mean_matrix = np.mean(matrices_array, axis=0)
            std_matrix = np.std(matrices_array, axis=0)
            
            # Calculate accuracy statistics
            accuracies = [metrics[key]['Accuracy'] * 100 for metrics in all_metrics]
            accuracy_mean = np.mean(accuracies)
            accuracy_std = np.std(accuracies)
            
            aggregated_cms[key] = {
                'mean': mean_matrix,
                'std': std_matrix,
                'accuracy_mean': accuracy_mean,
                'accuracy_std': accuracy_std,
                'classes': unique_classes[key]
            }
    
    return aggregated_cms

def plot_aggregated_confusion_matrices(aggregated_cms):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    
    title_mapping = {
        'CAD-RADS': 'Stenosis Severity',
        'Plaque Burden': 'Plaque Burden',
        'E': 'Modifier - E',
        'N': 'Modifier - N',
        'G': 'Modifier - G',
        'HRP': 'Modifier - HRP',
        'S': 'Modifier - S',
        'I': 'Modifier - I'
    }
    
    ordered_keys = ['CAD-RADS', 'Plaque Burden', 'E', 'N', 'G', 'HRP', 'S', 'I']
    
    ANNOT_FONTSIZE = 10
    LABEL_FONTSIZE = 10
    TITLE_FONTSIZE = 12
    
    for idx, key in enumerate(ordered_keys):
        if key in aggregated_cms:
            mean_matrix = aggregated_cms[key]['mean']
            std_matrix = aggregated_cms[key]['std']
            class_labels = aggregated_cms[key]['classes']
            
            # Create annotation text with mean ± std
            annot_matrix = np.array([[f"{mean:.1f}±{std:.1f}" 
                                    for mean, std in zip(row_mean, row_std)]
                                   for row_mean, row_std in zip(mean_matrix, std_matrix)])
            
            # Adjust font size for Stenosis Severity
            current_annot_fontsize = ANNOT_FONTSIZE - 3.5 if key == 'CAD-RADS' else ANNOT_FONTSIZE
            
            # Plot heatmap with mean values
            sns.heatmap(mean_matrix,
                       annot=annot_matrix,
                       fmt='',
                       cmap='Blues',
                       xticklabels=class_labels,
                       yticklabels=class_labels,
                       ax=axes[idx],
                       annot_kws={'size': current_annot_fontsize})
            
            axes[idx].set_title(
                f'{title_mapping[key]}\n'
                f'Accuracy: {aggregated_cms[key]["accuracy_mean"]:.1f}±{aggregated_cms[key]["accuracy_std"]:.1f}%',
                fontsize=TITLE_FONTSIZE
            )
            axes[idx].set_xlabel('Predicted', fontsize=LABEL_FONTSIZE)
            axes[idx].set_ylabel('True', fontsize=LABEL_FONTSIZE)
            axes[idx].tick_params(axis='both', labelsize=LABEL_FONTSIZE)
    
    plt.tight_layout()
    return fig

def filter_shot_files(result_files, n_shot=100):
    """Filter result files for specific n_shot"""
    filtered_files = []
    for file in result_files:
        shot_match = re.search(r'Shot(\d+)_', file)
        if shot_match and int(shot_match.group(1)) == n_shot:
            filtered_files.append(file)
    return filtered_files

def main():
    # Configuration
    DATASETS = ['InterTest', 'External']
    VERSION = {'InterTest': 'v1.1', 'External': 'v6.2'}
    RESULTS_DIR = {'InterTest': 'results', 'External': 'External_repeated/results'}
    N_SHOT = 100  # We only want shot=100 results
    
    for DATASET in DATASETS:
        print(f"\nProcessing {DATASET} dataset...")
        
        # Load ground truth data
        data_path = (f'sample_processed_internal_test_50_v1.2.csv' 
                    if DATASET == 'InterTest' 
                    else f'sample_processed_v6.4(외부병원).csv')
        data = pd.read_csv(data_path)
        data.fillna(value='None', inplace=True)
        labels = data.iloc[:,4:]
        
        # Get result files for shot=100 only
        pattern = f'result_1028_*_{DATASET}_{VERSION[DATASET]}_CoT_Shot*_Seed*.xlsx'
        all_files = glob(os.path.join(RESULTS_DIR[DATASET], pattern))
        result_files = filter_shot_files(all_files, N_SHOT)
        
        if not result_files:
            print(f"No result files found for {DATASET} with {N_SHOT}-shot")
            continue
        
        print(f"Found {len(result_files)} result files for {N_SHOT}-shot")
        
        # Get unique classes for each category
        unique_classes = get_unique_classes(result_files, labels)
        
        # Load and evaluate results
        all_metrics, all_cms = load_and_evaluate_results(result_files, labels, unique_classes)
        
        # Calculate aggregated confusion matrices
        aggregated_cms = calculate_aggregated_cms(all_metrics, all_cms, unique_classes)
        
        # Plot and save
        fig = plot_aggregated_confusion_matrices(aggregated_cms)
        output_filename = f'aggregated_confusion_matrices_{DATASET}_{N_SHOT}shot.png'
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved confusion matrix plot to {output_filename}")

if __name__ == "__main__":
    main()