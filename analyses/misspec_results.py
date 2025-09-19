import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

tensor = torch.tensor
array = np.array

def process_file_generic(file_path, columns):
    # Add header if missing
    with open(file_path, "r") as f:
        first_line = f.readline()
        if not first_line.startswith("Seed;RMSE History;MMD History;Log-Likelihood History;Deployment Time"):
            rest = f.read()
        else:
            rest = None

    if rest is not None:
        with open(file_path, "w") as f:
            f.write("Seed;RMSE History;MMD History;Log-Likelihood History;Deployment Time\n")
            f.write(first_line)
            f.write(rest)

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
        if column == "Max EIG History":
            sum_per_index = np.sum(arr_matrix, axis=1)
            mean_sum = np.mean(sum_per_index)
            stderr_sum = np.std(sum_per_index, ddof=1) / np.sqrt(len(sum_per_index))
            arrs.append(f"{round(mean_sum, 4)} ({round(stderr_sum, 4)})")
        else:
            mean_per_index = np.mean(arr_matrix, axis=0)
            #print(arr_matrix.shape)
            stderr_per_index = (np.std(arr_matrix, axis=0, ddof=1) / np.sqrt(arr_matrix.shape[0]))
            arrs.append(f"${round(mean_per_index[-1], 4)}\,({round(stderr_per_index[-1], 4)})$")
    print(file_path, " & ".join(arrs)) # Print summary statistics (mean and standard error of last experiment metric)
    return mean_per_index, stderr_per_index

def process_file(file_path):
    return process_file_generic(file_path, ["RMSE History"])

def process_file2(file_path):
    return process_file_generic(file_path, ["MMD History"])

def process_file3(file_path):
    return process_file_generic(file_path, ["Log-Likelihood History"])

file_info1 = [
    ("wellspecified_all/bayes_1.txt", "Bayes", "blue", "o"),        
    ("wellspecified_all/bayesw0.8_1.txt", "Bayes $\omega=0.8$", "orange", "s"), 
    ("wellspecified_all/defsm_1.txt", "Unweighted-SM", "red", "d"),
    ("wellspecified_all/laplante_1.txt", "Laplante", "gray", "^"),
    ("wellspecified_all/0.04_1.txt", "Exp-Decay $b = 0.04$", "deeppink", "h"),
    ("regression_random_0.04/wellspec/0.04random_1.txt", "Random + Exp-Decay", "gold", "P"),
    ("regression_beig_gibbsinf_0.04/wellspec/0.04beig_1.txt", "BEIG + Exp-Decay", "darkgreen", "v"),
]

file_info2 = [
    ("outlier_all/bayes_1.txt", "Bayes", "blue", "o"),
    ("outlier_all/bayesw0.8_1.txt", "Bayes $\omega=0.8$", "orange", "s"),
    ("outlier_all/defsm_1.txt", "Unweighted-SM", "red", "d"),
    ("outlier_all/laplante_1.txt", "Laplante", "gray", "^"),
    ("outlier_all/0.04_1.txt", "Exp-Decay $b = 0.04$", "deeppink", "h"),
    ("regression_random_0.04/outlier/0.04random_1.txt", "Random + Exp-Decay", "gold", "P"),
    ("regression_beig_gibbsinf_0.04/outlier/0.04beig_1.txt", "BEIG + Exp-Decay", "darkgreen", "v"),
]

rmse_results = []
mmd_results = []
loglike_results = []

for file_path, label, color, marker in file_info1:
    rmse_results.append((label, color, marker, *process_file(file_path)))
    mmd_results.append((label, color, marker, *process_file2(file_path)))
    loglike_results.append((label, color, marker, *process_file3(file_path)))

rmse_results2 = []
mmd_results2 = []
loglike_results2 = []

for file_path, label, color, marker in file_info2:
    rmse_results2.append((label, color, marker, *process_file(file_path)))
    mmd_results2.append((label, color, marker, *process_file2(file_path)))
    loglike_results2.append((label, color, marker, *process_file3(file_path)))

T = 10

import matplotlib.ticker as mticker

# Set up 2x3 grid of plots (top: well-specified, bottom: outlier)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Top row: Well-Specified Scenario
# RMSE plot
ax = axes[0, 0]
for label, color, marker, mean_per_index, stderr_per_index in rmse_results:
    ax.plot(mean_per_index, label=label, color=color, marker=marker)
    ax.fill_between(range(len(mean_per_index)), mean_per_index - stderr_per_index, mean_per_index + stderr_per_index, color=color, alpha=0.2)
ax.set_title("Mean RMSE (Well-Specified)")
ax.set_xlabel("Experiment")
ax.set_ylabel("RMSE")
ax.set_xticks(np.arange(1, T + 1, 3))
ax.legend()
ax.grid(True)

# MMD plot
ax = axes[0, 1]
for label, color, marker, mean_per_index, stderr_per_index in mmd_results:
    ax.plot(mean_per_index, label=label, color=color, marker=marker)
    ax.fill_between(range(len(mean_per_index)), mean_per_index - stderr_per_index, mean_per_index + stderr_per_index, color=color, alpha=0.2)
ax.set_title("Mean MMD (Well-Specified)")
ax.set_xlabel("Experiment")
ax.set_ylabel("MMD")
ax.set_xticks(np.arange(1, T + 1, 3))
ax.grid(True)

# Negative Log-Likelihood plot
ax = axes[0, 2]
for label, color, marker, mean_per_index, stderr_per_index in loglike_results:
    ax.plot(-mean_per_index, label=label, color=color, marker=marker)
    ax.fill_between(range(len(mean_per_index)), -mean_per_index - stderr_per_index, -mean_per_index + stderr_per_index, color=color, alpha=0.2)
ax.set_title("Mean NLL (Well-Specified)")
ax.set_xlabel("Experiment")
ax.set_ylabel("NLL")
ax.set_xticks(np.arange(1, T + 1, 3))
ax.grid(True)
inset_ax = ax.inset_axes([0.50, 0.41, 0.42, 0.44])
inset_ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
for label, color, marker, mean_per_index, stderr_per_index in loglike_results:
    n = len(mean_per_index)
    zoom_range = range(n-5, n)
    inset_ax.plot(zoom_range, -mean_per_index[-5:], color=color, marker=marker)
    inset_ax.fill_between(zoom_range, -mean_per_index[-5:] - stderr_per_index[-5:], -mean_per_index[-5:] + stderr_per_index[-5:], color=color, alpha=0.2)
inset_ax.set_xticks(zoom_range)
inset_ax.set_title("Zoomed In")
inset_ax.tick_params(axis="both", which="major", labelsize=8)
inset_ax.grid(True)
inset_ax.set_ylim(1.5, 3)

# Bottom row: Outlier Scenario
# RMSE plot
ax = axes[1, 0]
for label, color, marker, mean_per_index, stderr_per_index in rmse_results2:
    ax.plot(mean_per_index, label=label, color=color, marker=marker)
    ax.fill_between(range(len(mean_per_index)), mean_per_index - stderr_per_index, mean_per_index + stderr_per_index, color=color, alpha=0.2)
ax.set_title("Mean RMSE (Outlier)")
ax.set_xlabel("Experiment")
ax.set_ylabel("RMSE")
ax.set_xticks(np.arange(1, T + 1, 3))
ax.legend()
ax.grid(True)

# MMD plot
ax = axes[1, 1]
for label, color, marker, mean_per_index, stderr_per_index in mmd_results2:
    ax.plot(mean_per_index, label=label, color=color, marker=marker)
    ax.fill_between(range(len(mean_per_index)), mean_per_index - stderr_per_index, mean_per_index + stderr_per_index, color=color, alpha=0.2)
ax.set_title("Mean MMD (Outlier)")
ax.set_xlabel("Experiment")
ax.set_ylabel("MMD")
ax.set_xticks(np.arange(1, T + 1, 3))
ax.grid(True)

# Negative Log-Likelihood plot
ax = axes[1, 2]
for label, color, marker, mean_per_index, stderr_per_index in loglike_results2:
    ax.plot(-mean_per_index, label=label, color=color, marker=marker)
    ax.fill_between(range(len(mean_per_index)), -mean_per_index - stderr_per_index, -mean_per_index + stderr_per_index, color=color, alpha=0.2)
ax.set_title("Mean NLL (Outlier)")
ax.set_xlabel("Experiment")
ax.set_ylabel("NLL")
ax.set_xticks(np.arange(1, T + 1, 3))
ax.grid(True)
inset_ax = ax.inset_axes([0.50, 0.41, 0.42, 0.44])
inset_ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
for label, color, marker, mean_per_index, stderr_per_index in loglike_results2:
    n = len(mean_per_index)
    zoom_range = range(n-5, n)
    inset_ax.plot(zoom_range, -mean_per_index[-5:], color=color, marker=marker)
    inset_ax.fill_between(zoom_range, -mean_per_index[-5:] - stderr_per_index[-5:], -mean_per_index[-5:] + stderr_per_index[-5:], color=color, alpha=0.2)
inset_ax.set_xticks(zoom_range)
inset_ax.set_title("Zoomed In")
inset_ax.tick_params(axis="both", which="major", labelsize=8)
inset_ax.grid(True)
inset_ax.set_ylim(3, 6.5)

fig.suptitle("Well-Specified and Asymmetric Outlier Scenarios\nMean RMSE, MMD, and NLL (Std. Error over 90 Replications)", fontsize=14)
plt.tight_layout()
plt.savefig("combined_scenario_results.pdf", transparent=True, bbox_inches="tight")
plt.show()