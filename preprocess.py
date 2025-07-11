import pandas as pd
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy import stats

def process_cad_rads_labels(data_frame, label_column):
    valid_cad_rads = {'0', '1', '2', '3', '4A', '4B', '5'}
    valid_plaque_burden = {'None', 'P1', 'P2', 'P3', 'P4'}
    valid_modifiers = {'N', 'HRP', 'I', 'S', 'G', 'E'}

    def extract_data(label):
        label_str = str(label).strip().replace(' ', '')
        label_parts = label_str.split()
        if len(label_parts) > 1:
            label_str = label_parts[-1]

        parts = label_str.split('/')
        
        cadrads_val = parts[0] if parts[0] in valid_cad_rads else 'None'
        
        plaque_val = 'None'
        if len(parts) > 1:
            if parts[1] in valid_plaque_burden:
                plaque_val = parts[1]
            elif parts[1].startswith('P') and parts[1][1:] in {'1', '2', '3', '4'}:
                plaque_val = parts[1]
        
        modifiers_dict = {mod: 0 for mod in valid_modifiers}
        if len(parts) > 2 and parts[2].lower() != 'none':
            modifiers = re.findall(r'([A-Z]+)', parts[2])
            for mod in modifiers:
                if mod in valid_modifiers:
                    modifiers_dict[mod] = 1
                else:
                    for char in mod:
                        if char in valid_modifiers:
                            modifiers_dict[char] = 1

        return cadrads_val, plaque_val, modifiers_dict

    if isinstance(data_frame, pd.Series):
        data_frame = pd.DataFrame(data_frame)

    if isinstance(label_column, int):
        label_series = data_frame.iloc[:, label_column]
    else:
        label_series = data_frame[label_column]

    cadrads_list, plaque_list, modifiers_dicts = [], [], []
    for label in label_series:
        cadrads, plaque, mods_dict = extract_data(label)
        cadrads_list.append(cadrads)
        plaque_list.append(plaque)
        modifiers_dicts.append(mods_dict)

    result_df = pd.DataFrame()
    result_df['CAD-RADS'] = cadrads_list
    result_df['Plaque Burden'] = plaque_list
    for mod in valid_modifiers:
        result_df[mod] = [mods_dict[mod] for mods_dict in modifiers_dicts]

    return result_df

# def evaluate_performance(true_df, pred_df):
#     metrics = {}
#     columns = list(true_df.columns)
#     for column in columns:
#         if column in pred_df:
#             # NaN 값을 처리
#             true_values = true_df[column].fillna('None')
#             pred_values = pred_df[column].fillna('None')
            
#             if column == 'Plaque Burden':
#                 # S가 0이고 CAC_available이 1인 경우만 선택
#                 mask = (true_df['S'] == 0) & (true_df['CAC_available'] == 1)
#                 true_values = true_values[mask]
#                 pred_values = pred_values[mask]
#             elif column == 'CAD-RADS':
#                 # N이 0인 경우만 선택
#                 mask = (true_df['N'] == 0)
#                 true_values = true_values[mask]
#                 pred_values = pred_values[mask]
            
#             # 빈 배열 체크
#             if len(true_values) == 0 or len(pred_values) == 0:
#                 print(f"Warning: No valid samples for {column}. Skipping this column.")
#                 continue
            
#             # 정확도 계산
#             accuracy = accuracy_score(true_values, pred_values)
#             # F1 점수 계산
#             f1 = f1_score(true_values, pred_values, average='weighted')
#             # 혼동 행렬 계산
#             conf_mat = confusion_matrix(true_values, pred_values)

#             # 결과 저장
#             metrics[column] = {
#                 'Accuracy': accuracy,
#                 'F1 Score': f1,
#                 'Confusion Matrix': conf_mat
#             }

#     return metrics

def evaluate_performance(true_df, pred_df):
    metrics = {}
    columns = list(true_df.columns)
    
    for column in columns:
        if column in pred_df:
            # NaN 값을 처리
            true_values = true_df[column].fillna('None')
            pred_values = pred_df[column].fillna('None')
            
            # 특정 컬럼에 대한 마스킹 처리
            if column == 'Plaque Burden':
                mask = (true_df['S'] == 0) & (true_df['CAC_available'] == 1)
                true_values = true_values[mask]
                pred_values = pred_values[mask]
            elif column == 'CAD-RADS':
                mask = (true_df['N'] == 0)
                true_values = true_values[mask]
                pred_values = pred_values[mask]
            
            # 빈 배열 체크
            if len(true_values) == 0 or len(pred_values) == 0:
                print(f"Warning: No valid samples for {column}. Skipping this column.")
                continue
            
            # 데이터 타입 변환
            try:
                # 먼저 숫자로 변환 시도
                true_values = pd.to_numeric(true_values, errors='raise')
                pred_values = pd.to_numeric(pred_values, errors='raise')
            except:
                # 숫자 변환 실패시 문자열로 통일
                true_values = true_values.astype(str)
                pred_values = pred_values.astype(str)
            
            # 정확도 계산
            accuracy = accuracy_score(true_values, pred_values)
            # F1 점수 계산
            f1 = f1_score(true_values, pred_values, average='weighted')
            # 혼동 행렬 계산
            conf_mat = confusion_matrix(true_values, pred_values)

            # 결과 저장
            metrics[column] = {
                'Accuracy': accuracy,
                'F1 Score': f1,
                'Confusion Matrix': conf_mat
            }

    return metrics

# 혼동 행렬을 플롯하는 함수 추가
def plot_confusion_matrix(cm, class_labels, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()

def compare_certainty(labels, pred, certainty):
    results = {}
    
    for category in ['CAD-RADS', 'Plaque Burden']:
        if category == 'CAD-RADS':
            certainty_col = 'certainty_cadrads'
        else:  # Plaque Burden
            certainty_col = 'certainty_plaque'
        
        # 정답 여부 확인
        correct = labels[category] == pred[category]
        
        # 각 카테고리에 대한 마스크 생성
        if category == 'Plaque Burden':
            mask = (labels['S'] == 0) & (labels['CAC_available'] == 1)
        elif category == 'CAD-RADS':
            mask = (labels['N'] == 0)
        else:
            mask = pd.Series([True] * len(correct))
        
        # 마스크 적용
        correct = correct & mask
        
        # 정답인 경우와 아닌 경우의 certainty 분리
        certainty_correct = certainty[certainty_col][correct & mask]
        certainty_incorrect = certainty[certainty_col][~correct & mask]
        
        # 평균과 표준편차 계산
        avg_certainty_correct = certainty_correct.mean()
        std_certainty_correct = certainty_correct.std()
        avg_certainty_incorrect = certainty_incorrect.mean()
        std_certainty_incorrect = certainty_incorrect.std()
        
        # t-검정 수행
        t_statistic, p_value = stats.ttest_ind(certainty_correct, certainty_incorrect)
        
        results[category] = {
            'Avg Certainty (Correct)': avg_certainty_correct,
            'Std Certainty (Correct)': std_certainty_correct,
            'Avg Certainty (Incorrect)': avg_certainty_incorrect,
            'Std Certainty (Incorrect)': std_certainty_incorrect,
            'Difference': avg_certainty_correct - avg_certainty_incorrect,
            't-statistic': t_statistic,
            'p-value': p_value
        }
    
    return results

def make_many_shot_prompt(base_prompt, report, shots):
    """
    Create a prompt with n-shot examples
    
    Args:
        n (int): Number of shots to include
        report (str): The target report to classify
        shots (list): List of dictionaries containing shot examples with 'Report' and '0' columns
        
    Returns:
        str: Formatted prompt with shots
    """
    prompt = ""
    for i, shot in enumerate(shots):
        prompt += f'Example {i+1}:\n\n' + f"### Report:\n{shot['Report']}\n\n"
        prompt += f"### Rationale: \n{shot['CoT_from_claude']}\n===\n"
    
    prompt += f"""Now, extract the information from following report in the same format:

### Report: \n{report}\n\n### Rationale:\n"""
    
    prompt = base_prompt + '\n' + prompt
    return prompt