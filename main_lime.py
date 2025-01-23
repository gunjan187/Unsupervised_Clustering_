import numpy as np
import os
import pandas as pd
import json
import matplotlib.pyplot as plt  # For visualization
from lime import lime_tabular
from scluster import SCluster
from sklearn.metrics import silhouette_score
from sprtshap import *
from train_models import *
from helper import DatasetLoader
from sklearn.preprocessing import LabelEncoder
import time

# Set environment variables to avoid warnings and memory issues with KMeans
os.environ["OMP_NUM_THREADS"] = "2"  # Limit OpenMP threads to avoid KMeans issues
os.environ["LOKY_MAX_CPU_COUNT"] = str(12)  # Limit CPU usage for parallelism

# Class to perform clustering and explanations
class ClusteringAndExplanation:
    def __init__(self, clustering_type='kmeans', org=6, lim=10, stp=1):
        """
        Initialize the clustering and explanation pipeline.
        :param clustering_type: Clustering algorithm type (kmeans, hdbscan, meanshift)
        :param org: Minimum number of clusters for evaluation
        :param lim: Maximum number of clusters for evaluation
        :param stp: Step size for cluster range
        """
        self.clustering_type = clustering_type
        self.org = org
        self.lim = lim
        self.stp = stp

    def perform_clustering(self, data, dataset_name):
        """
        Apply clustering and return labels.
        :param data: Input data for clustering
        :param dataset_name: Name of the dataset being processed
        :return: Cluster labels
        """
        # Initialize the clustering algorithm with parameters
        clusterer = SCluster(typ=self.clustering_type, org=self.org, lim=self.lim, stp=self.stp)
        clusterer.fit(data)  # Fit the clustering algorithm to the data
        labels = clusterer.labels_  # Get the labels assigned to each data point

        # Save clustering labels to a CSV file
        labels_output_path = f"./Results/{dataset_name}_clustering_labels_{self.clustering_type}.csv"
        os.makedirs(os.path.dirname(labels_output_path), exist_ok=True)
        pd.DataFrame({'Cluster_Label': labels}).to_csv(labels_output_path, index=False)
        print(f"Clustering labels saved to: {labels_output_path}")

        return labels

    def visualize_clusters(self, data, cluster_labels, dataset_name, explanations=None, method='lime'):
        """
        Visualize clusters with optional annotations based on explanations.
        :param data: Input data points
        :param cluster_labels: Cluster labels for each data point
        :param dataset_name: Name of the dataset being processed
        :param explanations: Feature explanations for each cluster (optional)
        :param method: Explanation method (e.g., lime)
        """
        plt.figure(figsize=(10, 8))
        unique_clusters = np.unique(cluster_labels)  # Get unique cluster IDs

        # Calculate the centroid for each cluster
        cluster_centroids = np.array([data[cluster_labels == c].mean(axis=0) for c in unique_clusters])

        for cluster, centroid in zip(unique_clusters, cluster_centroids):
            cluster_data = data[cluster_labels == cluster]  # Get data points belonging to the current cluster
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster}')
            plt.scatter(centroid[0], centroid[1], s=200, c='black', marker='X')  # Mark the centroid

            # Annotate clusters with the top feature if explanations are provided
            if explanations:
                top_features = explanations.get(str(cluster), [])
                if top_features and isinstance(top_features, list) and len(top_features) > 0:
                    top_feature_info = top_features[0]  # Get the top feature's information
                    if isinstance(top_feature_info, tuple) and len(top_feature_info) == 2:
                        top_feature_name, top_feature_importance = top_feature_info
                        if isinstance(top_feature_importance, (float, int)):
                            # Annotate the centroid with the top feature name and its importance
                            plt.annotate(f'{top_feature_name} ({top_feature_importance:.2f})',
                                         xy=(centroid[0], centroid[1]),
                                         fontsize=10, ha='center', color='red')

        plt.title(f'Clusters Visualization with {method.capitalize()} Explanations ({dataset_name})')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()

        # Save the visualization as a PNG file
        visualization_output_path = f'./Results/{dataset_name}_cluster_visualization_{method}.png'
        plt.savefig(visualization_output_path)
        plt.show()
        print(f"Visualization saved to: {visualization_output_path}")

    def explain_clusters(self, data, cluster_labels, dataset_name, method='lime', model=None, num_features=2, max_samples=500):
        """
        Explain clusters using the selected method (LIME or KernelShap).
        :param data: Input data points
        :param cluster_labels: Cluster labels for each data point
        :param dataset_name: Name of the dataset being processed
        :param method: Explanation method (e.g., lime)
        :param model: Model for prediction (if required by explanation method)
        :param num_features: Number of top features to include in the explanation
        :param max_samples: Maximum number of samples to explain per cluster
        :return: Dictionary of explanations for each cluster
        """
        explanations = {}
        cluster_centroids = np.array([data[cluster_labels == c].mean(axis=0) for c in np.unique(cluster_labels)])

        for cluster_id in np.unique(cluster_labels):
            cluster_data = data[cluster_labels == cluster_id]  # Get data points for the current cluster
            print(f"Explaining cluster {cluster_id} with {len(cluster_data)} samples.")

            # Randomly sample data points if the cluster size exceeds max_samples
            if len(cluster_data) > max_samples:
                print(f"Sampling {max_samples} from {len(cluster_data)} samples for cluster {cluster_id}.")
                cluster_data = cluster_data[np.random.choice(len(cluster_data), max_samples, replace=False)]

            if method == 'lime':
                # Initialize the LIME explainer
                explainer = lime_tabular.LimeTabularExplainer(cluster_data, discretize_continuous=False)

                def cluster_distance_predict(data):
                    """
                    Predict cluster probabilities based on distances to centroids.
                    :param data: Input data points
                    :return: Probability of belonging to each cluster
                    """
                    data = np.asarray(data, dtype=np.float64)  # Ensure numeric input
                    centroids = np.asarray(cluster_centroids, dtype=np.float64)  # Ensure numeric centroids

                    # Calculate distances and probabilities
                    distances = np.linalg.norm(data[:, None] - centroids[None, :], axis=2)
                    probabilities = 1 / (distances + 1e-10)  # Avoid division by zero
                    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
                    return probabilities

                # Generate explanations for each instance in the cluster
                explanations[str(cluster_id)] = [
                    explainer.explain_instance(instance, cluster_distance_predict, num_features=num_features).as_list()
                    for instance in cluster_data
                ]

            else:
                raise ValueError("Unsupported explanation method")

        # Save explanations to a JSON file
        explanations_output_path = f"./Results/{dataset_name}_explanations_{method}.json"
        os.makedirs(os.path.dirname(explanations_output_path), exist_ok=True)
        with open(explanations_output_path, 'w') as f:
            json.dump(explanations, f, indent=4)
        print(f"Explanations saved to: {explanations_output_path}")

        return explanations

def preprocess_data(data):
    """
    Preprocess the data to ensure it is suitable for clustering.
    :param data: Raw input data (Pandas DataFrame)
    :return: Preprocessed data (NumPy array)
    """
    # Identify non-numeric columns
    non_numeric_cols = data.select_dtypes(include=['object', 'category']).columns
    if len(non_numeric_cols) > 0:
        print(f"Encoding non-numeric columns: {list(non_numeric_cols)}")
        encoder = LabelEncoder()
        for col in non_numeric_cols:
            data[col] = encoder.fit_transform(data[col])

    return data.values  # Return the preprocessed data as a NumPy array

# Load data using DatasetLoader
loader = DatasetLoader(data_folder="./Data")

# List of datasets to process
datasets = ["bank", "brca", "census", "credit"]

for dataset_name in datasets:
    start_time = time.time()
    try:
        print(f"Processing dataset: {dataset_name}")

        # Load dataset
        data = loader.load_dataset(dataset_name)

        # Preprocess the data
        X_train = preprocess_data(data)

        # Initialize clustering and explanation pipeline
        pipeline = ClusteringAndExplanation(clustering_type='kmeans', org=6, lim=10, stp=1)

        # Perform clustering
        cluster_labels = pipeline.perform_clustering(X_train, dataset_name)

        # Train model (optional for some explanation methods)
        model = None  # Add model training if necessary

        # Explain clusters using LIME
        explanations = pipeline.explain_clusters(X_train, cluster_labels, dataset_name, method='lime', model=model, max_samples=500)

        # Visualize clusters with LIME explanations
        pipeline.visualize_clusters(X_train, cluster_labels, dataset_name, explanations, method='lime')

    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {e}")

    end_time = time.time()
    print(f"Finished processing dataset: {dataset_name} in {end_time - start_time:.2f} seconds.")
