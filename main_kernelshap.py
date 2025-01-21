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
from load_data import *

# Set environment variables to avoid warnings and memory issues with KMeans
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["LOKY_MAX_CPU_COUNT"] = "12"


# Class to perform clustering and explanations
class ClusteringAndExplanation:
    def __init__(self, clustering_type='kmeans', org=6, lim=10, stp=1):
        self.clustering_type = clustering_type
        self.org = org
        self.lim = lim
        self.stp = stp

    def perform_clustering(self, data):
        """
        Apply clustering and return labels.
        """
        clusterer = SCluster(typ=self.clustering_type, org=self.org, lim=self.lim, stp=self.stp)
        clusterer.fit(data)
        labels = clusterer.labels_

        # Save clustering labels
        labels_output_path = f"./Results/clustering_labels_{self.clustering_type}.csv"
        os.makedirs(os.path.dirname(labels_output_path), exist_ok=True)
        pd.DataFrame({'Cluster_Label': labels}).to_csv(labels_output_path, index=False)
        print(f"Clustering labels saved to: {labels_output_path}")

        return labels

    def visualize_clusters(self, data, cluster_labels, explanations=None, method='kernelshap'):
        """
        Visualize clusters with optional annotations based on explanations.
        :param data: Input data for clustering
        :param cluster_labels: Cluster labels assigned to the data
        :param explanations: Dictionary of explanations for each cluster (optional)
        :param method: The explainability method used
        """
        plt.figure(figsize=(10, 8))
        unique_clusters = np.unique(cluster_labels)
        for cluster in unique_clusters:
            cluster_data = data[cluster_labels == cluster]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster}')
            
            # Annotate clusters with the top feature if explanations are provided
            if explanations:
                top_features = explanations.get(str(cluster), [])
                if len(top_features) > 0:
                    # Ensure top_features is a list
                    if isinstance(top_features, np.ndarray):
                        top_features = top_features.tolist()

                    # Get the top feature name and importance
                    top_feature_name = f"Feature {np.argmax(np.abs(top_features)) + 1}"
                    top_feature_importance = np.max(np.abs(top_features))

                    # Annotate the cluster center with feature info
                    plt.annotate(f'{top_feature_name} ({top_feature_importance:.2f})',
                                xy=(np.mean(cluster_data[:, 0]), np.mean(cluster_data[:, 1])),
                                fontsize=10, ha='center', bbox=dict(boxstyle="round", fc="w", alpha=0.8))

        plt.title(f'Clusters Visualization with {method.capitalize()} Explanations')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        visualization_output_path = f'./Results/cluster_visualization_{method}.png'
        os.makedirs(os.path.dirname(visualization_output_path), exist_ok=True)
        plt.savefig(visualization_output_path)
        plt.show()
        print(f"Visualization saved to: {visualization_output_path}")

    def explain_clusters(self, data, cluster_labels, method='kernelshap', model=None, num_features=3):
        """
        Explain clusters using the selected method (LIME, RankShap, or KernelShap).
        """
        explanations = {}
        cluster_centroids = np.array([data[cluster_labels == c].mean(axis=0) for c in np.unique(cluster_labels)])

        for cluster_id in np.unique(cluster_labels):
            cluster_data = data[cluster_labels == cluster_id]

            # Skip empty clusters
            if cluster_data.shape[0] == 0:
                print(f"Cluster {cluster_id} is empty, skipping explanation.")
                continue

            # Ensure cluster_data is 2D
            if cluster_data.ndim == 1:
                cluster_data = cluster_data.reshape(1, -1)

            print(f"Explaining cluster {cluster_id} with {len(cluster_data)} samples.")

            try:
                if method == 'lime':
                    explainer = lime_tabular.LimeTabularExplainer(cluster_data, discretize_continuous=False)

                    def cluster_distance_predict(data):
                        distances = np.linalg.norm(data[:, None] - cluster_centroids[None, :], axis=2)
                        probabilities = 1 / (distances + 1e-10)
                        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
                        return probabilities

                    explanations[str(cluster_id)] = [
                        explainer.explain_instance(instance, cluster_distance_predict, num_features=num_features).as_list()
                        for instance in cluster_data
                    ]

                elif method == 'kernelshap':
                    if model:
                        explanations[str(cluster_id)] = kernelshap(model, cluster_data, num_features)
                    else:
                        def dummy_kernel_predict(data):
                            probabilities = np.random.rand(data.shape[0], 2)
                            probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
                            return probabilities

                        explanations[str(cluster_id)] = kernelshap(dummy_kernel_predict, cluster_data, num_features)

                else:
                    raise ValueError("Unsupported explanation method")
            except Exception as e:
                print(f"Failed to explain cluster {cluster_id}: {e}")

        # Convert all NumPy arrays to lists for JSON serialization
        explanations_serializable = {
            cluster_id: (explanation.tolist() if isinstance(explanation, np.ndarray) else explanation)
            for cluster_id, explanation in explanations.items()
        }

        explanations_output_path = f"./Results/explanations_{method}.json"
        os.makedirs(os.path.dirname(explanations_output_path), exist_ok=True)
        with open(explanations_output_path, 'w') as f:
            json.dump(explanations_serializable, f, indent=4)
        print(f"Explanations saved to: {explanations_output_path}")

        return explanations


# Load data
X_train, y_train, X_test, y_test, mapping_dict = load_data("./Data", "unsupervised_dataset")

# Initialize clustering and explanation pipeline
pipeline = ClusteringAndExplanation(clustering_type='kmeans', org=6, lim=10, stp=1)

# Perform clustering
cluster_labels = pipeline.perform_clustering(X_train)

# Train model (optional for some explanation methods)
model = None
if y_train is not None:
    model = train_model(X_train, y_train, "nn")

# Explain clusters using KernelSHAP
explanations = pipeline.explain_clusters(X_train, cluster_labels, method='kernelshap', model=model)

# Visualize clusters with KernelSHAP explanations
pipeline.visualize_clusters(X_train, cluster_labels, explanations, method='kernelshap')
