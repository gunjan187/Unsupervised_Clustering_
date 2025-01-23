# Updated SCluster Class with Preprocessing and Error Handling
from sklearn.cluster import KMeans, MeanShift
from sklearn.metrics import silhouette_score
from hdbscan import HDBSCAN
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

CLUSTERING = {
    'kmeans': lambda df, k: KMeans(n_clusters=k).fit(df).labels_,
    'hdbscan': lambda df, s: HDBSCAN(min_cluster_size=s).fit(df).labels_,
    'meanshift': lambda df, q: MeanShift(bandwidth=0.025 * min(q, 50)).fit(df).labels_
}

class SCluster:
    def __init__(self, typ='kmeans', org=2, lim=20, stp=1, dup=0.95, max_retries=3):
        self.type = typ
        self.org = org
        self.lim = lim + 1
        self.stp = stp
        self.dup = dup
        self.max_retries = max_retries
        self.function = CLUSTERING[self.type.lower()]
        self.max = -1
        self.scores = {}
        self.labels_ = []

    def preprocess_data(self, data):
        """
        Preprocess the data to ensure it is numeric.
        :param data: Input data (Pandas DataFrame or NumPy array)
        :return: Preprocessed NumPy array
        """
        if isinstance(data, pd.DataFrame):
            non_numeric_cols = data.select_dtypes(include=['object', 'category']).columns
            if len(non_numeric_cols) > 0:
                print(f"Encoding non-numeric columns: {list(non_numeric_cols)}")
                encoder = LabelEncoder()
                for col in non_numeric_cols:
                    data[col] = encoder.fit_transform(data[col])
            return data.values
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError("Input data must be a Pandas DataFrame or NumPy array.")

    def adapt_silhouette(self, labels):
        data, labels = self.df[labels > -1], labels[labels > -1]
        if data.shape[0] == 0:
            return -1
        retries = 0
        while retries < self.max_retries:
            try:
                return silhouette_score(data, labels, sample_size=self.size) * (labels.shape[0] / self.n)
            except Exception as e:
                retries += 1
                self.size = int(self.size * self.dup)
                print(f"Silhouette score calculation failed. Retrying... ({retries}/{self.max_retries})")
        print("Silhouette score calculation failed after maximum retries.")
        return -1

    def fit(self, data):
        self.n = data.shape[0]
        self.size = self.n
        self.df = self.preprocess_data(data)
        for i in range(self.org, self.lim, self.stp):
            try:
                label = self.function(self.df, i)
                silho = self.adapt_silhouette(label)
                self.scores[silho] = label
                self.max = silho if self.max < silho else self.max
                print(f"cluster kind: {self.type}, input value = {i}, silhouette = {round(silho, 2)}")
            except Exception as e:
                print(f"Clustering failed for input value = {i}: {e}")
        self.labels_ = self.scores[self.max]
        return self
