import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

tensor = torch.tensor
array = np.array

def process_file_generic(file_path, columns):
    df = pd.read_csv(file_path, sep=";")
    arrs = []
    for column in columns:
        if column not in df.columns:
            print(f"Column '{column}' not found in {file_path}. Skipping.")
            continue
        if column == "Deployment Time":
            arr_lists = df[column].apply(np.array)
            arr_matrix = np.vstack(arr_lists.values)
            print(arr_matrix.shape)
        else:
            arr_lists = df[column].apply(eval).apply(np.array)
            arr_matrix = np.vstack(arr_lists.values)
        #if column == "Max EIG History":
        #    sum_per_index = np.sum(arr_matrix, axis=1)
        #    mean_sum = np.mean(sum_per_index)
        #    stderr_sum = np.std(sum_per_index, ddof=1) / np.sqrt(len(sum_per_index))
        #    arrs.append(f"{round(mean_sum, 4)} ({round(stderr_sum, 4)})")
        #else:
        cum_arr_matrix = np.cumsum(arr_matrix, axis=1)
        mean_per_index = np.mean(cum_arr_matrix, axis=0)
        stderr_per_index = np.std(cum_arr_matrix, axis=0, ddof=1) / np.sqrt(cum_arr_matrix.shape[0])
        #arrs.append(f"{round(mean_per_index[-1], 4)} ({round(stderr_per_index[-1], 4)})")
    #print(file_path, " & ".join(arrs))
    return mean_per_index, stderr_per_index

def process_file(file_path, columns=["Log-Likelihood History"]):
    return process_file_generic(file_path, columns)

file_info_wellspec = [
    ("location_wellspecv4_closer_param_d=2_paramupd2_ucb12_w0.2/beig_bayesinf/_loc_find_results_neg-log_misspec_none_N10000_M100_assum_std0.5_w0.2_1.txt", "Bayes", "blue", "o"),  
    ("location_wellspecv4_closer_param_d=2_paramupd2_ucb12_w0.2/geig_gibbsinf_defsm/_loc_find_results_score-matching-default_misspec_none_N10000_M100_assum_std0.5_w0.2_1.txt", "Unweighted-SM", "red", "s"),  
    ("location_wellspecv4_closer_param_d=2_paramupd2_ucb12_w0.2/geig_gibbsinf_laplante/_loc_find_results_score-matching-weighted_misspec_none_N10000_M100_assum_std0.5_w0.2_1.txt", "Laplante", "grey", "d"),  
    ("location_wellspecv4_closer_param_d=2_paramupd2_ucb12_w0.2/geig_gibbsinf_weighted-sm-expdec-0.04/_loc_find_results_score-matching-weighted_misspec_none_N10000_M100_assum_std0.5_w0.2_1.txt", "Exp-Decay $b = 0.04$", "deeppink", "^"),  
    ("location_wellspecv4_closer_param_d=2_paramupd2_ucb12_w0.2/geig_gibbsinf_weighted-sm-expdec-0.06/_loc_find_results_score-matching-weighted_misspec_none_N10000_M100_assum_std0.5_w0.2_1.txt", "Exp-Decay $b = 0.06$", "orange", "v"),  
    ("location_wellspecv4_closer_param_d=2_paramupd2_ucb12_w0.2/geig_gibbsinf_weighted-sm-expdec-0.10/_loc_find_results_score-matching-weighted_misspec_none_N10000_M100_assum_std0.5_w0.2_1.txt", "Exp-Decay $b = 0.10$", "green", "*"),  
]

file_info_outlier = [
    ("location_outlierv4_closer_param_d=2_paramupd2_ucb12_w0.2/beig_bayesinf/_loc_find_results_neg-log_misspec_outlier_N10000_M100_assum_std0.5_w0.2_1.txt", "Bayes", "blue", "o"),  
    ("location_outlierv4_closer_param_d=2_paramupd2_ucb12_w0.2/geig_gibbsinf_defsm/_loc_find_results_score-matching-default_misspec_outlier_N10000_M100_assum_std0.5_w0.2_1.txt", "Unweighted-SM", "red", "s"),  
    ("location_outlierv4_closer_param_d=2_paramupd2_ucb12_w0.2/geig_gibbsinf_laplante/_loc_find_results_score-matching-weighted_misspec_outlier_N10000_M100_assum_std0.5_w0.2_1.txt", "Laplante", "grey", "d"),  
    ("location_outlierv4_closer_param_d=2_paramupd2_ucb12_w0.2/geig_gibbsinf_weighted-sm-expdec-0.04/_loc_find_results_score-matching-weighted_misspec_outlier_N10000_M100_assum_std0.5_w0.2_1.txt", "Exp-Decay $b = 0.04$", "deeppink", "^"),  
    ("location_outlierv4_closer_param_d=2_paramupd2_ucb12_w0.2/geig_gibbsinf_weighted-sm-expdec-0.06/_loc_find_results_score-matching-weighted_misspec_outlier_N10000_M100_assum_std0.5_w0.2_1.txt", "Exp-Decay $b = 0.06$", "orange", "v"),  
    ("location_outlierv4_closer_param_d=2_paramupd2_ucb12_w0.2/geig_gibbsinf_weighted-sm-expdec-0.10/_loc_find_results_score-matching-weighted_misspec_outlier_N10000_M100_assum_std0.5_w0.2_1.txt", "Exp-Decay $b = 0.10$", "green", "*"),  
]

def get_max_eig_results(file_info):
    results = []
    for file_path, label, color, marker in file_info:
        mean_per_index, stderr_per_index = process_file(file_path, ["Max EIG History"])
        results.append((label, color, marker, mean_per_index, stderr_per_index))
    return results

max_eig_results_wellspec = get_max_eig_results(file_info_wellspec)
max_eig_results_outlier = get_max_eig_results(file_info_outlier)

# Plot side by side
fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
x = np.arange(1, 31)

# Wellspec subplot
ax = axes[0]
for label, color, marker, mean_per_index, stderr_per_index in max_eig_results_wellspec:
    ax.plot(x, mean_per_index, label=label, color=color, marker=marker)
    ax.fill_between(x, mean_per_index - stderr_per_index, mean_per_index + stderr_per_index, color=color, alpha=0.2)
ax.set_title("Well-Specified Scenario Results\nMean Cumulative EIG (Std. Error over 100 Replications)")
ax.set_xlabel("Experiment")
ax.set_ylabel("Cumulative EIG")
ax.set_xticks(list(np.arange(1, 31, 5)) + [30])
ax.set_yticks(np.arange(0, 42, 5))
ax.legend()
ax.grid()

# Outlier subplot
ax = axes[1]
for label, color, marker, mean_per_index, stderr_per_index in max_eig_results_outlier:
    ax.plot(x, mean_per_index, label=label, color=color, marker=marker)
    ax.fill_between(x, mean_per_index - stderr_per_index, mean_per_index + stderr_per_index, color=color, alpha=0.2)
ax.set_title("Asymmetric Outlier Scenario Results\nMean Cumulative EIG (Std. Error over 100 Replications)")
ax.set_xlabel("Experiment")
ax.set_xticks(list(np.arange(1, 31, 5)) + [30])
#ax.legend()
ax.grid()

plt.tight_layout()
plt.savefig("location_experiment_results_comparison.pdf", transparent=True, bbox_inches="tight")
plt.show()
