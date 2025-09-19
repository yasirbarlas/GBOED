import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import io

tensor = torch.tensor
array = np.array

def process_file_generic(file_path, columns):
    # Add header if missing (added header depends on what experimental design problem was run, remove Max EIG History for regression)
    with open(file_path, "r") as f:
        first_line = f.readline()
        if not first_line.startswith("Seed;RMSE History;MMD History;Log-Likelihood History;Max EIG History;Deployment Time"):
            rest = f.read()
        else:
            rest = None

    if rest is not None:
        with open(file_path, "w") as f:
            f.write("Seed;RMSE History;MMD History;Log-Likelihood History;Max EIG History;Deployment Time\n")
            f.write(first_line)
            f.write(rest)

    # The below lets you interpret results for seeds meeting a certain threshold, useful to see how behaviour changes as replications increases (e.g. seeds from 50 to 1500)
    # Read the file and split into separate dataframes for each set
    dfs = []
    with open(file_path, "r") as f:
        lines = f.readlines()
    header = lines[0]
    data_lines = lines[1:]
    # Find indices where seed == 50 (start of each set)
    start_indices = [i for i, line in enumerate(data_lines) if line.startswith("50;")]
    start_indices.append(len(data_lines))
    for idx in range(len(start_indices) - 1):
        chunk = data_lines[start_indices[idx]:start_indices[idx+1]]
        df_chunk = pd.read_csv(io.StringIO(header + "".join(chunk)), sep=";")
        # Remove seeds above threshold if there are any (e.g. 5000)
        df_chunk = df_chunk[df_chunk["Seed"] <= 5000]
        dfs.append(df_chunk)
    # Concatenate back
    df = pd.concat(dfs, ignore_index=True)
    
    arrs = []
    for column in columns:
        if column not in df.columns:
            print(f"Column '{column}' not found in {file_path}. Skipping.")
            continue
        if column == "Deployment Time":
            arr_lists = df[column].apply(np.array)
            arr_matrix = np.vstack(arr_lists.values)
            #print(arr_matrix.shape)
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
            if column == "Log-Likelihood History":
                arrs.append(f"${round(-mean_per_index[-1], 3)}\,({round(stderr_per_index[-1], 3)})$")
            else:
                arrs.append(f"${round(mean_per_index[-1], 3)}\,({round(stderr_per_index[-1], 3)})$")
    print(file_path, " & ".join(arrs))

def process_file(file_path, columns=["Log-Likelihood History"]):
    return process_file_generic(file_path, columns)

# Find the seed with the highest final log-likelihood in a given file
def find_seed_with_highest_loglike(file_path):
    df = pd.read_csv(file_path, sep=";")
    if "Log-Likelihood History" not in df.columns or "Seed" not in df.columns:
        print(f"Required columns not found in {file_path}.")
        return None
    loglike_histories = df["Log-Likelihood History"].apply(eval).apply(np.array)
    arr_matrix = np.vstack(loglike_histories.values)
    final_loglikes = arr_matrix[:, -1]
    min_idx = np.argmax(final_loglikes)
    min_seed = df.iloc[min_idx]["Seed"]
    min_value = final_loglikes[min_idx]
    print(f"Seed with highest final log-likelihood in {file_path}: {min_seed} (value: {min_value})")
    return min_seed, min_value

# Plot histograms of final log-likelihoods for all replications, separate histogram for each file input
def plot_loglike_histograms(file_paths, labels=["A", "B"], colors=None, bins=50, xlim=(1, 2)):
    # Collect all final_loglikes to determine global bin edges
    all_loglikes = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, sep=";")
        if "Log-Likelihood History" not in df.columns:
            continue
        loglike_histories = df["Log-Likelihood History"].apply(eval).apply(np.array)
        arr_matrix = np.vstack(loglike_histories.values)
        final_loglikes = -arr_matrix[:, -1]
        all_loglikes.append(final_loglikes)
    # Flatten all_loglikes and compute bin edges
    all_loglikes_flat = np.concatenate(all_loglikes)
    bin_edges = np.linspace(xlim[0], xlim[1], bins + 1)
    plt.figure(figsize=(8, 6))
    for idx, final_loglikes in enumerate(all_loglikes):
        label = labels[idx] if labels is not None else file_paths[idx]
        color = colors[idx] if colors is not None else None
        plt.hist(final_loglikes, bins=bin_edges, alpha=0.5, label=label, color=color, edgecolor='black')
    plt.xlim(*xlim)
    plt.xlabel("NLL")
    plt.ylabel("Frequency")
    plt.title("Histogram of NLLs From 100 Replications")
    if labels is not None:
        plt.legend()
    plt.tight_layout()
    plt.savefig("loglike_histograms.pdf", transparent=True, bbox_inches='tight')
    plt.show()

pharmaco_wellspec = [
    ("pharma_wellspec_param/beig_bayesinf/_pharmaco_results_neg-log_misspec_none_N10000_M100_w0.4_1.txt", "$\\omega=0.1$", "blue"),  
    ("pharma_wellspec_param/beig_bayesinf_w0.4/_pharmaco_results_neg-log_misspec_none_N10000_M100_w0.4_1.txt", "$\\omega=0.1$", "blue"),
    ("pharma_wellspec_param/geig_gibbsinf_defsm/_pharmaco_results_score-matching-default_misspec_none_N10000_M100_w0.4_1.txt", "$\\omega=0.1$", "blue"),  
    ("pharma_wellspec_param/geig_gibbsinf_laplante/_pharmaco_results_score-matching-weighted_misspec_none_N10000_M100_w0.4_1.txt", "$\\omega=0.1$", "blue"),  
    ("pharma_wellspec_param/geig_gibbsinf_weighted-sm-expdec-0.08/_pharmaco_results_score-matching-weighted_misspec_none_N10000_M100_w0.4_1.txt", "$\\omega=0.1$", "blue"),      
    ("pharma_wellspec_param/geig_gibbsinf_weighted-sm-expdec-0.12/_pharmaco_results_score-matching-weighted_misspec_none_N10000_M100_w0.4_1.txt", "$\\omega=0.1$", "blue"),  
    ("pharma_wellspec_param/geig_gibbsinf_weighted-sm-expdec-0.16/_pharmaco_results_score-matching-weighted_misspec_none_N10000_M100_w0.4_1.txt", "$\\omega=0.1$", "blue"),  
    ("pharma_wellspec_param/geig_gibbsinf_weighted-sm-expdec-0.20/_pharmaco_results_score-matching-weighted_misspec_none_N10000_M100_w0.4_1.txt", "$\\omega=0.1$", "blue"),  
    ("pharma_wellspec_param/beig_gibbsinf_laplante/_pharmaco_results_score-matching-weighted_misspec_none_N10000_M100_w0.4_1.txt", "$\\omega=0.1$", "blue"),
    ("pharma_wellspec_param/random_gibbsinf_laplante/_pharmaco_results_score-matching-weighted_misspec_none_N10000_M100_w0.4_1.txt", "$\\omega=0.1$", "orange"),
]

pharmaco_outlier = [
    ("pharma_outlier_param/beig_bayesinf/_pharmaco_results_neg-log_misspec_outlier_N10000_M100_w0.4_1.txt", "$\\omega=0.1$", "blue"),  
    ("pharma_outlier_param/beig_bayesinf_w0.4/_pharmaco_results_neg-log_misspec_outlier_N10000_M100_w0.4_1.txt", "$\\omega=0.1$", "blue"),
    ("pharma_outlier_param/geig_gibbsinf_defsm/_pharmaco_results_score-matching-default_misspec_outlier_N10000_M100_w0.4_1.txt", "$\\omega=0.1$", "blue"),  
    ("pharma_outlier_param/geig_gibbsinf_laplante/_pharmaco_results_score-matching-weighted_misspec_outlier_N10000_M100_w0.4_1.txt", "$\\omega=0.1$", "blue"),  
    ("pharma_outlier_param/geig_gibbsinf_weighted-sm-expdec-0.08/_pharmaco_results_score-matching-weighted_misspec_outlier_N10000_M100_w0.4_1.txt", "$\\omega=0.1$", "blue"),  
    ("pharma_outlier_param/geig_gibbsinf_weighted-sm-expdec-0.12/_pharmaco_results_score-matching-weighted_misspec_outlier_N10000_M100_w0.4_1.txt", "$\\omega=0.1$", "blue"),  
    ("pharma_outlier_param/geig_gibbsinf_weighted-sm-expdec-0.16/_pharmaco_results_score-matching-weighted_misspec_outlier_N10000_M100_w0.4_1.txt", "$\\omega=0.1$", "blue"),  
    ("pharma_outlier_param/geig_gibbsinf_weighted-sm-expdec-0.20/_pharmaco_results_score-matching-weighted_misspec_outlier_N10000_M100_w0.4_1.txt", "$\\omega=0.1$", "blue"),  
    ("pharma_outlier_param/beig_gibbsinf_laplante/_pharmaco_results_score-matching-weighted_misspec_outlier_N10000_M100_w0.4_1.txt", "$\\omega=0.1$", "blue"),
    ("pharma_outlier_param/random_gibbsinf_laplante/_pharmaco_results_score-matching-weighted_misspec_outlier_N10000_M100_w0.4_1.txt", "$\\omega=0.1$", "orange"),
]

pharmaco_errordist = [
    ("pharma_errordist_param/beig_bayesinf/_pharmaco_results_neg-log_misspec_error-dist_N10000_M100_w0.4_1.txt", "$\\omega=0.1$", "blue"),  
    ("pharma_errordist_param/beig_bayesinf_w0.4/_pharmaco_results_neg-log_misspec_error-dist_N10000_M100_w0.4_1.txt", "$\\omega=0.1$", "blue"),
    ("pharma_errordist_param/geig_gibbsinf_defsm/_pharmaco_results_score-matching-default_misspec_error-dist_N10000_M100_w0.4_1.txt", "$\\omega=0.1$", "blue"),  
    ("pharma_errordist_param/geig_gibbsinf_laplante/_pharmaco_results_score-matching-weighted_misspec_error-dist_N10000_M100_w0.4_1.txt", "$\\omega=0.1$", "blue"),  
    ("pharma_errordist_param/geig_gibbsinf_weighted-sm-expdec-0.08/_pharmaco_results_score-matching-weighted_misspec_error-dist_N10000_M100_w0.4_1.txt", "$\\omega=0.1$", "blue"),  
    ("pharma_errordist_param/geig_gibbsinf_weighted-sm-expdec-0.12/_pharmaco_results_score-matching-weighted_misspec_error-dist_N10000_M100_w0.4_1.txt", "$\\omega=0.1$", "blue"),  
    ("pharma_errordist_param/geig_gibbsinf_weighted-sm-expdec-0.16/_pharmaco_results_score-matching-weighted_misspec_error-dist_N10000_M100_w0.4_1.txt", "$\\omega=0.1$", "blue"),  
    ("pharma_errordist_param/geig_gibbsinf_weighted-sm-expdec-0.20/_pharmaco_results_score-matching-weighted_misspec_error-dist_N10000_M100_w0.4_1.txt", "$\\omega=0.1$", "blue"),  
    ("pharma_errordist_param/beig_gibbsinf_laplante/_pharmaco_results_score-matching-weighted_misspec_error-dist_N10000_M100_w0.4_1.txt", "$\\omega=0.1$", "blue"),
    ("pharma_errordist_param/random_gibbsinf_laplante/_pharmaco_results_score-matching-weighted_misspec_error-dist_N10000_M100_w0.4_1.txt", "$\\omega=0.1$", "orange"),
]

# Choose method(s) of analysis
def filer(file_info):
    for file_path, label, color in file_info:
        process_file(file_path, ["RMSE History", "MMD History", "Log-Likelihood History"])
        #plot_loglike_histogram(file_path)
        #find_seed_with_highest_loglike(file_path)

# Run analysis for chosen sets of files
for i in [pharmaco_wellspec, pharmaco_outlier, pharmaco_errordist]:#, infor4, infor5, infor6]:#[file_info1, file_info2, file_info4, file_info5]:
    filer(i)
    print("\n")