import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

tensor = torch.tensor
array = np.array

# Uncomment the set you want to analyse (one set is for well-specified, the other for misspecified)
# The reason why there are 3 files per method is that each represents a different model (we averaged over 3 different true models in the paper, so we do the same here)
# One can instead have a single file with all of the models collated together

bayes_files = [
    ("tableeig_wellspec_1/v2bayeseig_bayes/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
    ("tableeig_wellspec_2/v2bayeseig_bayes/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
    ("tableeig_wellspec_3/v2bayeseig_bayes/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
]

def_sm_files = [
    ("tableeig_wellspec_1/v2bayeseig_defsm/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
    ("tableeig_wellspec_2/v2bayeseig_defsm/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
    ("tableeig_wellspec_3/v2bayeseig_defsm/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
]

v2bayeseig_expdec0_04_files = [
    ("tableeig_wellspec_1/v2bayeseig_expdec0.04/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
    ("tableeig_wellspec_2/v2bayeseig_expdec0.04/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
    ("tableeig_wellspec_3/v2bayeseig_expdec0.04/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
]

v2bayeseig_expdec0_08_files = [
    ("tableeig_wellspec_1/v2bayeseig_expdec0.08/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
    ("tableeig_wellspec_2/v2bayeseig_expdec0.08/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
    ("tableeig_wellspec_3/v2bayeseig_expdec0.08/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
]

v2bayeseig_expdec0_10_files = [
    ("tableeig_wellspec_1/v2bayeseig_expdec0.10/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
    ("tableeig_wellspec_2/v2bayeseig_expdec0.10/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
    ("tableeig_wellspec_3/v2bayeseig_expdec0.10/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
]

v2bayeseig_laplante_files = [
    ("tableeig_wellspec_1/v2bayeseig_laplante/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
    ("tableeig_wellspec_2/v2bayeseig_laplante/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
    ("tableeig_wellspec_3/v2bayeseig_laplante/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
]

v2bayeseig_c2_files = [
    ("tableeig_wellspec_1/v2bayeseig_c2/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
    ("tableeig_wellspec_2/v2bayeseig_c2/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
    ("tableeig_wellspec_3/v2bayeseig_c2/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
]

v2bayeseig_c10_files = [
    ("tableeig_wellspec_1/v2bayeseig_c10/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
    ("tableeig_wellspec_2/v2bayeseig_c10/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
    ("tableeig_wellspec_3/v2bayeseig_c10/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
]

w0_8_files = [
    ("tableeig_wellspec_1/v2bayeseig_bayesw0.8/bayeseig_1.txt", "$\\omega=0.8$", "orange"),
    ("tableeig_wellspec_2/v2bayeseig_bayesw0.8/bayeseig_1.txt", "$\\omega=0.8$", "orange"),
    ("tableeig_wellspec_3/v2bayeseig_bayesw0.8/bayeseig_1.txt", "$\\omega=0.8$", "orange"),
]

# bayes_files = [
#     ("tableeig_outlier_1/v2bayeseig_bayes/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
#     ("tableeig_outlier_2/v2bayeseig_bayes/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
#     ("tableeig_outlier_3/v2bayeseig_bayes/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
# ]

# def_sm_files = [
#     ("tableeig_outlier_1/v2bayeseig_defsm/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
#     ("tableeig_outlier_2/v2bayeseig_defsm/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
#     ("tableeig_outlier_3/v2bayeseig_defsm/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
# ]

# v2bayeseig_expdec0_04_files = [
#     ("tableeig_outlier_1/v2bayeseig_expdec0.04/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
#     ("tableeig_outlier_2/v2bayeseig_expdec0.04/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
#     ("tableeig_outlier_3/v2bayeseig_expdec0.04/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
# ]

# v2bayeseig_expdec0_08_files = [
#     ("tableeig_outlier_1/v2bayeseig_expdec0.08/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
#     ("tableeig_outlier_2/v2bayeseig_expdec0.08/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
#     ("tableeig_outlier_3/v2bayeseig_expdec0.08/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
# ]

# v2bayeseig_expdec0_10_files = [
#     ("tableeig_outlier_1/v2bayeseig_expdec0.10/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
#     ("tableeig_outlier_2/v2bayeseig_expdec0.10/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
#     ("tableeig_outlier_3/v2bayeseig_expdec0.10/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
# ]

# v2bayeseig_laplante_files = [
#     ("tableeig_outlier_1/v2bayeseig_laplante/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
#     ("tableeig_outlier_2/v2bayeseig_laplante/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
#     ("tableeig_outlier_3/v2bayeseig_laplante/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
# ]

# v2bayeseig_c2_files = [
#     ("tableeig_outlier_1/v2bayeseig_c2/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
#     ("tableeig_outlier_2/v2bayeseig_c2/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
#     ("tableeig_outlier_3/v2bayeseig_c2/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
# ]

# v2bayeseig_c10_files = [
#     ("tableeig_outlier_1/v2bayeseig_c10/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
#     ("tableeig_outlier_2/v2bayeseig_c10/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
#     ("tableeig_outlier_3/v2bayeseig_c10/bayeseig_1.txt", "$\\omega=0.1$", "blue"),
# ]

# w0_8_files = [
#     ("tableeig_outlier_1/v2bayeseig_bayesw0.8/bayeseig_1.txt", "$\\omega=0.8$", "orange"),
#     ("tableeig_outlier_2/v2bayeseig_bayesw0.8/bayeseig_1.txt", "$\\omega=0.8$", "orange"),
#     ("tableeig_outlier_3/v2bayeseig_bayesw0.8/bayeseig_1.txt", "$\\omega=0.8$", "orange"),
# ]

# Combine results for each variable into a pandas DataFrame
def combine_files_to_df(file_list):
    dfs = []
    # Read headers from the first file
    first_file_path, label, color = file_list[0]

    # Add header if missing
    with open(first_file_path, "r") as f:
        first_line = f.readline()
        if not first_line.startswith("Seed;RMSE History;MMD History;Log-Likelihood History;Table EIG"):
            rest = f.read()
        else:
            rest = None

    if rest is not None:
        with open(first_file_path, "w") as f:
            f.write("Seed;RMSE History;MMD History;Log-Likelihood History;Table EIG\n")
            f.write(first_line)
            f.write(rest)

    df = pd.read_csv(first_file_path, sep=";")
    
    df["label"] = label
    df["color"] = color
    dfs.append(df)
    # Use the same headers for the rest
    columns = df.columns
    for file_path, label, color in file_list[1:]:
        df = pd.read_csv(file_path, sep=";", header=None, names=columns)
        df["label"] = label
        df["color"] = color
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df

bayes_df = combine_files_to_df(bayes_files)
w0_8_df = combine_files_to_df(w0_8_files)
def_sm_df = combine_files_to_df(def_sm_files)
v2bayeseig_expdec0_04_df = combine_files_to_df(v2bayeseig_expdec0_04_files)
v2bayeseig_expdec0_08_df = combine_files_to_df(v2bayeseig_expdec0_08_files)
v2bayeseig_expdec0_10_df = combine_files_to_df(v2bayeseig_expdec0_10_files)
v2bayeseig_laplante_df = combine_files_to_df(v2bayeseig_laplante_files)
v2bayeseig_c2_df = combine_files_to_df(v2bayeseig_c2_files)
v2bayeseig_c10_df = combine_files_to_df(v2bayeseig_c10_files)

def process_df_diff(df, bayes_df, column):
    # Subtract each value in df[column] from corresponding value in bayes_df[column]
    arr_lists = df[column].apply(eval).apply(lambda x: np.array(x[0]))
    bayes_arr_lists = bayes_df[column].apply(eval).apply(lambda x: np.array(x[0]))
    assert len(arr_lists) == len(bayes_arr_lists), "Arrays must be of the same length"
    diff_lists = [arr - bayes_arr for arr, bayes_arr in zip(arr_lists, bayes_arr_lists)]
    arr_matrix = np.sum(np.vstack(diff_lists), axis=1)
    print(arr_matrix.shape)
    mean_per_index = np.mean(arr_matrix, axis=0)
    stderr_per_index = np.std(arr_matrix, axis=0, ddof=1) / np.sqrt(arr_matrix.shape[0])
    print(f"${np.round(mean_per_index, 4)}\,({np.round(stderr_per_index, 4)})$")
    return mean_per_index, stderr_per_index

def process_df_diff_tableeig(df, bayes_df):
    return process_df_diff(df, bayes_df, "Table EIG")

process_df_diff_tableeig(bayes_df, bayes_df) # This will be zero
process_df_diff_tableeig(w0_8_df, bayes_df)
process_df_diff_tableeig(def_sm_df, bayes_df)
process_df_diff_tableeig(v2bayeseig_laplante_df, bayes_df)
process_df_diff_tableeig(v2bayeseig_c2_df, bayes_df)
process_df_diff_tableeig(v2bayeseig_c10_df, bayes_df)
process_df_diff_tableeig(v2bayeseig_expdec0_04_df, bayes_df)
#process_df_diff_tableeig(v2bayeseig_expdec0_08_df, bayes_df)
#process_df_diff_tableeig(v2bayeseig_expdec0_10_df, bayes_df)