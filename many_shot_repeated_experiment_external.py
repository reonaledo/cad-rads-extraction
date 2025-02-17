import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import os
import sys
import signal
import psutil
import gc
from datetime import datetime
from preprocess import process_cad_rads_labels, evaluate_performance, plot_confusion_matrix, compare_certainty, make_many_shot_prompt

# Configuration
API = 'gemini'
USE_COT = True
USE_CERTAINTY = False
USE_MANYSHOT = True
VERSION = 'v6.2'

# Experiment parameters
N_SHOTS = [10, 20, 50, 100]
SEEDS = [421, 422, 423, 424, 425, 426, 427, 428, 429]
# N_SHOTS = [3]
# SEEDS = [425, 426, 427, 428, 429]

# Safety and logging configurations
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
API_CALL_DELAY = 1  # seconds between API calls
MEMORY_THRESHOLD = 90  # percentage
MAX_API_ERRORS = 5  # Maximum consecutive API errors before stopping
BASE_DIR = "External_repeated/"
LOG_DIR = BASE_DIR + "experiment_logs"
CHECKPOINT_DIR = BASE_DIR +"checkpoints"
RESULTS_DIR = BASE_DIR +"results"

def setup_signal_handlers():
    """Setup graceful shutdown handlers"""
    def signal_handler(signum, frame):
        print("\nReceived shutdown signal. Completing current experiment before exit...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def check_system_resources():
    """Check if system has enough resources to continue"""
    memory = psutil.virtual_memory()
    if memory.percent > MEMORY_THRESHOLD:
        raise ResourceWarning(f"Memory usage too high: {memory.percent}%")
    
    disk = psutil.disk_usage('/')
    if disk.percent > MEMORY_THRESHOLD:
        raise ResourceWarning(f"Disk usage too high: {disk.percent}%")

def setup_directories():
    """Create necessary directories for logs and checkpoints"""
    for directory in [LOG_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)

def load_checkpoint():
    """Load the latest checkpoint if it exists"""
    checkpoint_file = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.csv")
    if os.path.exists(checkpoint_file):
        return pd.read_csv(checkpoint_file)
    return None

def save_checkpoint(completed_experiments):
    """Save current progress to checkpoint with backup"""
    checkpoint_file = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.csv")
    backup_file = os.path.join(CHECKPOINT_DIR, f"checkpoint_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    # Save to both main checkpoint and backup
    pd.DataFrame(completed_experiments).to_csv(checkpoint_file, index=False)
    pd.DataFrame(completed_experiments).to_csv(backup_file, index=False)

def log_error(n_shot, seed, error_msg):
    """Log errors to file with detailed information"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    error_file = os.path.join(LOG_DIR, f"error_log_{timestamp}.txt")
    
    with open(error_file, "a", encoding='utf-8') as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"N_SHOT: {n_shot}\n")
        f.write(f"SEED: {seed}\n")
        f.write(f"Error: {error_msg}\n")
        f.write("="*50 + "\n")

def validate_data_files():
    """Validate that all required data files exist and are readable"""
    required_files = {
        f'sample_processed_{VERSION}(외부병원).csv': "test data",
        'manyshot_pool_all_from_claude.csv': "manyshot pool",
        "prompt_system_1016_cot_edit.txt": "system prompt",
        "prompt_user_few_cot_ManyShot.txt": "user prompt"
    }
    
    for file_path, description in required_files.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing {description} file: {file_path}")
        
        # Try reading each file to ensure they're not corrupted
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read()
        except Exception as e:
            raise IOError(f"Error reading {description} file {file_path}: {str(e)}")

def validate_response(response, trigger):
    """Validate that the response is properly formatted"""
    if not isinstance(response, str):
        raise ValueError("Response is not a string")
    
    if not response.strip():
        raise ValueError("Response is empty")
    
    if trigger not in response:
        raise ValueError(f"Response missing trigger: {trigger}")
    
    parts = response.split(trigger)
    if len(parts) != 2:
        raise ValueError("Malformed response structure")
    
    return True

def run_experiment(N_SHOT, SEED):
    MAX_OUTPUT_LENGTH = 1000 if USE_COT else 50
    consecutive_api_errors = 0
    
    # Gemini API setup
    import google.generativeai as genai
    api_key = 'YOUR_API_KEY'
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    genai.configure(api_key=api_key)
    
    # Load and validate data
    data = pd.read_csv(f'sample_processed_{VERSION}(외부병원).csv')
    if data.empty:
        raise ValueError("Empty test data file")
    data.fillna(value='None', inplace=True)
    
    # Load prompts
    with open("prompt_system_1016_cot_edit.txt", 'r', encoding='utf-8') as f:
        prompt_sys = f.read()
    
    with open("prompt_user_few_cot_ManyShot.txt", 'r') as f:
        prompt_user_few = f.read()
    
    if not prompt_sys.strip() or not prompt_user_few.strip():
        raise ValueError("Empty prompt file")
    
    # Initialize Gemini client
    client = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config={
            "temperature": 0,
            "max_output_tokens": MAX_OUTPUT_LENGTH,
            "response_mime_type": "text/plain",
        },
        system_instruction=prompt_sys
    )
    
    def get_response(prompt, retry_count=0): 
        """Get response with retry logic and error handling"""
        nonlocal consecutive_api_errors
        
        try:
            check_system_resources()
            chat_session = client.start_chat()
            response = chat_session.send_message(prompt)
            time.sleep(API_CALL_DELAY)
            
            response_text = response.text
            validate_response(response_text, trigger)
            
            consecutive_api_errors = 0  # Reset counter on success
            return response_text
            
        except Exception as e:
            consecutive_api_errors += 1
            if consecutive_api_errors >= MAX_API_ERRORS:
                raise Exception(f"Too many consecutive API errors ({MAX_API_ERRORS}): {str(e)}")
            
            if retry_count < MAX_RETRIES:
                time.sleep(RETRY_DELAY * (retry_count + 1))  # Exponential backoff
                return get_response(prompt, retry_count + 1)
            raise e
    
    trigger = "Final Answer (CAD-RADS/Plaque Burden/Modifier):"
    manyshot_pool = pd.read_csv('manyshot_pool_all_from_claude.csv')
    responses = []
    labels = []
    
    # Verify manyshot pool
    if len(manyshot_pool) < N_SHOT:
        raise ValueError(f"Manyshot pool has {len(manyshot_pool)} samples, but N_SHOT is {N_SHOT}")
    
    # Main experiment loop
    experiment_start_time = time.time()
    for i in tqdm(range(len(data)), desc=f"N_SHOT={N_SHOT}, SEED={SEED}"):
        sampled_shots = manyshot_pool.sample(n=N_SHOT, random_state=SEED)[['Report', 'CoT_from_claude']].to_dict('records')
        
        prompt = make_many_shot_prompt(
            prompt_user_few,
            report=data['Report'][i],
            shots=sampled_shots
        )
        
        try:
            response = get_response(prompt)
            
            if trigger in response:
                responses.append(response)
            else:
                print(f'Response {i} does not contain the final answer. Retrying.')
                new_response = get_response(prompt + "\n\n### Rationale:\n" + response + "\n\n" + trigger)
                responses.append(response + "\n\n" + trigger + new_response)
                
        except Exception as e:
            log_error(N_SHOT, SEED, f"Error in sample {i}: {str(e)}")
            raise e
        
        # Periodic garbage collection
        if i % 10 == 0:
            gc.collect()
    
    experiment_duration = time.time() - experiment_start_time
    
    # Process results
    try:
        labels = [r.split(trigger)[-1].split('\n')[0] for r in responses]
        labels = pd.DataFrame(labels)
        responses = pd.DataFrame(responses)
        labels = process_cad_rads_labels(labels, 0)
    except Exception as e:
        log_error(N_SHOT, SEED, f"Error processing results: {str(e)}")
        raise e
    
    # Save results with metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    TAG = f"CoT_Shot{N_SHOT}_Seed{SEED}"
    filename = os.path.join(RESULTS_DIR, f'result_1028_{API}_External_{VERSION}_{TAG}_{timestamp}.xlsx')
    
    try:
        result_df = pd.concat([data, labels, responses], axis=1)
        
        # Add metadata
        metadata = pd.DataFrame([{
            'experiment_timestamp': timestamp,
            'n_shot': N_SHOT,
            'seed': SEED,
            'duration_seconds': experiment_duration,
            'api': API,
            'version': VERSION
        }])
        
        # Save results and metadata
        with pd.ExcelWriter(filename) as writer:
            result_df.to_excel(writer, sheet_name='Results', index=False)
            metadata.to_excel(writer, sheet_name='Metadata', index=False)
        
        print(f"Results saved to: {filename}")
        
    except Exception as e:
        log_error(N_SHOT, SEED, f"Error saving results: {str(e)}")
        raise e
    
    return filename

def main():
    print("Starting experiment suite...")
    print(f"Total combinations to test: {len(N_SHOTS) * len(SEEDS)}")
    
    setup_signal_handlers()
    setup_directories()
    
    # Validate environment
    try:
        validate_data_files()
        check_system_resources()
    except Exception as e:
        print(f"Initialization Error: {str(e)}")
        return
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    completed_experiments = [] if checkpoint is None else checkpoint.to_dict('records')
    
    # Create a set of completed experiments
    completed_pairs = {(exp['N_SHOT'], exp['SEED']) for exp in completed_experiments}
    
    # Calculate progress
    total_experiments = len(N_SHOTS) * len(SEEDS)
    completed_count = len(completed_experiments)
    
    print(f"\nExperiment Progress: {completed_count}/{total_experiments} completed")
    print(f"Estimated remaining time: {(total_experiments - completed_count) * 25} minutes (approximate)")
    
    # Run remaining experiments
    for n_shot in N_SHOTS:
        for seed in SEEDS:
            if (n_shot, seed) in completed_pairs:
                print(f"Skipping completed experiment: N_SHOT={n_shot}, SEED={seed}")
                continue
            
            print(f"\nStarting experiment: N_SHOT={n_shot}, SEED={seed}")
            experiment_start = time.time()
            
            try:
                filename = run_experiment(n_shot, seed)
                duration = time.time() - experiment_start
                
                result = {
                    'N_SHOT': n_shot,
                    'SEED': seed,
                    'filename': filename,
                    'status': 'success',
                    'duration': duration,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
            except Exception as e:
                error_msg = str(e)
                print(f"Error in experiment N_SHOT={n_shot}, SEED={seed}: {error_msg}")
                
                result = {
                    'N_SHOT': n_shot,
                    'SEED': seed,
                    'filename': None,
                    'status': f'failed: {error_msg}',
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            
            completed_experiments.append(result)
            save_checkpoint(completed_experiments)
            
            # Update summary
            summary_df = pd.DataFrame(completed_experiments)
            summary_df.to_csv('experiment_summary.csv', index=False)
            
            # Clear memory
            gc.collect()
    
    # Final summary
    print("\nExperiment suite completed!")
    success_count = sum(1 for exp in completed_experiments if exp['status'] == 'success')
    failed_count = total_experiments - success_count
    
    print("\nFinal Statistics:")
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Success rate: {(success_count/total_experiments)*100:.2f}%")
    
    if failed_count > 0:
        print("\nFailed experiments:")
        for exp in completed_experiments:
            if 'failed' in exp['status']:
                print(f"N_SHOT={exp['N_SHOT']}, SEED={exp['SEED']}: {exp['status']}")

if __name__ == "__main__":
    main()