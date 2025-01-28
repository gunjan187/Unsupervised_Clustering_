import numpy as np
import os
import pandas as pd
import json
import matplotlib.pyplot as plt  # For visualization
from scluster import SCluster
from sklearn.metrics import silhouette_score
from sprtshap import kernelshap
from train_models import train_model
from helper import DatasetLoader
from sklearn.preprocessing import LabelEncoder
import time

# Set environment variables to avoid warnings and memory issues with KMeans
os.environ["OMP_NUM_THREADS"] = "2"  # Limit OpenMP threads to avoid KMeans issues
os.environ["LOKY_MAX_CPU_COUNT"] = "12"  # Limit CPU usage for parallelism

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
        clusterer = SCluster(typ=self.clustering_type, org=self.org, lim=self.lim, stp=self.stp)
        clusterer.fit(data)  # Fit the clustering algorithm to the data
        labels = clusterer.labels_  # Get the labels assigned to each data point

        # Save clustering labels to a CSV file
        labels_output_path = f"./Results/{dataset_name}_clustering_labels_kernelshap.csv"
        os.makedirs(os.path.dirname(labels_output_path), exist_ok=True)
        pd.DataFrame({'Cluster_Label': labels}).to_csv(labels_output_path, index=False)
        print(f"Clustering labels saved to: {labels_output_path}")

        return labels

    def visualize_clusters(self, data, cluster_labels, dataset_name, explanations=None, method='kernelshap'):
        """
        Visualize clusters with optional annotations based on explanations.
        """
        plt.figure(figsize=(10, 8))  # Set the figure size for the plot
        unique_clusters = np.unique(cluster_labels)  # Get unique cluster IDs

        # Calculate the centroid for each cluster
        cluster_centroids = np.array([data[cluster_labels == c].mean(axis=0) for c in unique_clusters])

        for cluster, centroid in zip(unique_clusters, cluster_centroids):
            cluster_data = data[cluster_labels == cluster]  # Get data points belonging to the current cluster
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster}')  # Plot cluster points
            plt.scatter(centroid[0], centroid[1], s=200, c='black', marker='X')  # Mark the centroid

            # Annotate clusters with the top feature if explanations are provided
            if explanations and str(cluster) in explanations:
                top_features = explanations[str(cluster)]
                if isinstance(top_features, list) and len(top_features) > 0:
                    top_feature_name = f"Feature {np.argmax(np.abs(top_features)) + 1}"
                    top_feature_importance = np.max(np.abs(top_features))
                    plt.annotate(
                        f'{top_feature_name} ({top_feature_importance:.2f})',
                        xy=(centroid[0], centroid[1]),
                        fontsize=10,
                        ha='center',
                        color='red'
                    )

        plt.title(f'Clusters Visualization with {method.capitalize()} Explanations ({dataset_name})')  # Set plot title
        plt.xlabel('Feature 1')  # Label for x-axis
        plt.ylabel('Feature 2')  # Label for y-axis
        plt.legend()  # Add a legend to the plot

        # Save the visualization as a PNG file
        visualization_output_path = f'./Results/{dataset_name}_cluster_visualization_kernelshap.png'
        os.makedirs(os.path.dirname(visualization_output_path), exist_ok=True)
        plt.savefig(visualization_output_path)
        plt.show()  # Display the plot
        print(f"Visualization saved to: {visualization_output_path}")

    def explain_clusters(self, data, cluster_labels, dataset_name, method='kernelshap', model=None, num_features=3, max_samples=500):
        """
        Explain clusters using KernelSHAP.
        """
        explanations = {}  # Dictionary to store explanations for each cluster
        for cluster_id in np.unique(cluster_labels):
            cluster_data = data[cluster_labels == cluster_id]  # Get data points for the current cluster
            print(f"Explaining cluster {cluster_id} with {len(cluster_data)} samples.")

            if len(cluster_data) > max_samples:
                print(f"Sampling {max_samples} from {len(cluster_data)} samples for cluster {cluster_id}.")
                cluster_data = cluster_data[np.random.choice(len(cluster_data), max_samples, replace=False)]

            try:
                if method == 'kernelshap':
                    if model:
                        explanations[str(cluster_id)] = kernelshap(model, cluster_data, num_features)
                    else:
                        def dummy_kernel_predict(data):
                            probabilities = np.random.rand(data.shape[0], 2)  # Generate random probabilities
                            probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)  # Normalize
                            return probabilities

                        explanations[str(cluster_id)] = kernelshap(dummy_kernel_predict, cluster_data, num_features)
                else:
                    raise ValueError("Unsupported explanation method.")
            except Exception as e:
                print(f"Error explaining cluster {cluster_id}: {e}")

        explanations_serializable = {
            cluster_id: explanation.tolist() if isinstance(explanation, np.ndarray) else explanation
            for cluster_id, explanation in explanations.items()
        }

        explanations_output_path = f"./Results/{dataset_name}_explanations_kernelshap.json"
        os.makedirs(os.path.dirname(explanations_output_path), exist_ok=True)
        with open(explanations_output_path, 'w') as f:
            json.dump(explanations_serializable, f, indent=4)
        print(f"Explanations saved to: {explanations_output_path}")

        return explanations

def preprocess_data(data):
    """
    Preprocess the data to ensure it is suitable for clustering.
    """
    non_numeric_cols = data.select_dtypes(include=['object', 'category']).columns
    if len(non_numeric_cols) > 0:
        print(f"Encoding non-numeric columns: {list(non_numeric_cols)}")
        encoder = LabelEncoder()
        for col in non_numeric_cols:
            data[col] = encoder.fit_transform(data[col])
    return data.values

# Load data using DatasetLoader
loader = DatasetLoader(data_folder="./Data")

datasets = ["bank", "brca", "census", "credit"]

for dataset_name in datasets:
    start_time = time.time()
    try:
        print(f"Processing dataset: {dataset_name}")

        data = loader.load_dataset(dataset_name)
        X_train = preprocess_data(data)

        pipeline = ClusteringAndExplanation(clustering_type='kmeans', org=6, lim=10, stp=1)
        cluster_labels = pipeline.perform_clustering(X_train, dataset_name)

        model = None

        explanations = pipeline.explain_clusters(X_train, cluster_labels, dataset_name, method='kernelshap', model=model, max_samples=500)

        pipeline.visualize_clusters(X_train, cluster_labels, dataset_name, explanations, method='kernelshap')

    except Exception as e:
        print(f"Error processing dataset {dataset_name}: {e}")

    end_time = time.time()
    print(f"Finished processing dataset: {dataset_name} in {end_time - start_time:.2f} seconds.")
