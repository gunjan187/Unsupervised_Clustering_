import numpy as np


def map_S(S, mapping_dict):
    '''
    Maps a subset of feature indices to the corresponding subset of columns.
    mapping_dict contains feature indices as keys and their corresponding columns as values.
    '''

    S_cols_list = [mapping_dict[i] for i in S]
    S_cols = sorted([item for sublist in S_cols_list for item in sublist])
    return S_cols


def get_ranking(shap_vals, abs=True):
    # Indices with biggest to smallest (absolute) SHAP value
    if abs:
        return np.argsort(np.abs(shap_vals))[::-1]
    else:
        return np.argsort(shap_vals)[::-1]


def mode_rows(a):
    a = np.ascontiguousarray(a)
    void_dt = np.dtype((np.void, a.dtype.itemsize * np.prod(a.shape[1:])))
    _,ids, count = np.unique(a.view(void_dt).ravel(), \
                                return_index=1,return_counts=1)
    largest_count_id = ids[count.argmax()]
    most_frequent_row = a[largest_count_id]
    return most_frequent_row


def calc_fwer(top_K):
    most_common_row = mode_rows(top_K)
    fwer = 1 - np.mean(np.all(np.array(top_K)==most_common_row,axis=1))
    return np.round(fwer, 3)# if Round else fwer
    
def welch_df(var1, var2, n1, n2, var_of_mean=False):
    if var_of_mean:
        num = (var1 + var2)**2
        denom = var1**2/(n1-1) + var2**2/(n2-1)
    else:
        num = (var1/n1 + var2/n2)**2
        denom = (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)
    df = num / denom
    return df

def shap_vals_to_ranks(shap_vals, abs=True):
    N_pts, N_runs, d = shap_vals.shape
    shap_ranks = np.array([get_ranking(shap_vals[i,j,:], abs=abs) for i in range(N_pts) for j in range(N_runs)]).reshape(shap_vals.shape)
    return shap_ranks

import os
import pandas as pd

class DatasetLoader:
    def __init__(self, data_folder="Data"):
        """
        Initializes the DatasetLoader with the specified data folder.
        :param data_folder: Path to the folder containing datasets.
        """
        self.data_folder = data_folder
        self.datasets = {
            "bank": "bank.csv",
            "brca": "brca_small.csv",
            "census": "census_X.csv",
            "credit": "credit.csv"
        }

    def load_dataset(self, dataset_name):
        """
        Loads the specified dataset by name.
        :param dataset_name: The key name of the dataset (e.g., "bank", "brca").
        :return: A pandas DataFrame of the dataset.
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {list(self.datasets.keys())}")

        dataset_path = os.path.join(self.data_folder, self.datasets[dataset_name])

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        print(f"Loading dataset: {dataset_name} from {dataset_path}")
        return pd.read_csv(dataset_path)

# Example Usage
if __name__ == "__main__":
    loader = DatasetLoader()

    # Example: Load the "bank" dataset
    try:
        bank_data = loader.load_dataset("bank")
        print(bank_data.head())
    except Exception as e:
        print(str(e))

    # Example: Load the "credit" dataset
    try:
        credit_data = loader.load_dataset("credit")
        print(credit_data.head())
    except Exception as e:
        print(str(e))
