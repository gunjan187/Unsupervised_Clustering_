import numpy as np
import os
import pandas as pd
import json
import matplotlib.pyplot as plt  # For visualization
from lime import lime_tabular
from scluster import SCluster
from sklearn.metrics import silhouette_score
from rankshap import *
from sprtshap import *
from train_models import *
from load_data import *

# Set environment variables to avoid warnings and memory issues with KMeans
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["LOKY_MAX_CPU_COUNT"] = str(12)

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

    def perform_clustering(self, data):
        """
        Apply clustering and return labels.
        :param data: Input data for clustering
        :return: Cluster labels
        """
        clusterer = SCluster(typ=self.clustering_type, org=self.org, lim=self.lim, stp=self.stp)
        clusterer.fit(data)
        labels = clusterer.labels_

        # Ensure Results folder exists
        labels_output_path = "./Results/clustering_labels_lime.csv"
        os.makedirs(os.path.dirname(labels_output_path), exist_ok=True)
        
        # Remove the file if it already exists
        if os.path.exists(labels_output_path):
            os.remove(labels_output_path)

        # Save clustering labels
        pd.DataFrame({'Cluster_Label': labels}).to_csv(labels_output_path, index=False)
        print(f"Clustering labels saved to: {labels_output_path}")

        return labels

    def visualize_clusters(self, data, cluster_labels, explanations=None):
        """
        Visualize clusters with optional annotations based on explanations.
        :param data: Input data for clustering
        :param cluster_labels: Cluster labels assigned to the data
        :param explanations: Dictionary of explanations for each cluster (optional)
        """
        import matplotlib.pyplot as plt
        import numpy as np

        plt.figure(figsize=(8, 6))
        unique_clusters = np.unique(cluster_labels)
        for cluster in unique_clusters:
            cluster_data = data[cluster_labels == cluster]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster}')
            
            # Annotate clusters with the top feature, if explanations are provided
            if explanations:
                top_feature = explanations.get(str(cluster), [])
                if top_feature:
                    top_feature_name, top_feature_importance = top_feature[0]  # Get the most important feature
                    plt.annotate(f'{top_feature_name} ({top_feature_importance:.2f})', 
                                xy=(np.mean(cluster_data[:, 0]), np.mean(cluster_data[:, 1])),
                                fontsize=10, ha='center')

        plt.title('Clusters Visualization with Explanations')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.savefig('./Results/cluster_visualization_lime.png')
        plt.show()

    def explain_clusters(self, data, cluster_labels, method='lime', model=None, num_features=3):
        """
        Explain clusters using the selected method (LIME, RankShap, or KernelShap).
        """
        explanations = {}
        for cluster_id in np.unique(cluster_labels):
            cluster_data = data[cluster_labels == cluster_id]
            print(f"Explaining cluster {cluster_id} with {len(cluster_data)} samples.")

            if method == 'lime':
                explainer = lime_tabular.LimeTabularExplainer(
                    cluster_data, discretize_continuous=False
                )

                # Define a dummy model if no actual model exists
                def dummy_predict(data):
                    n_samples = data.shape[0]
                    return np.hstack([np.ones((n_samples, 1)) * 0.5, np.ones((n_samples, 1)) * 0.5])

                explanations[str(cluster_id)] = [
                    explainer.explain_instance(instance, dummy_predict, num_features=num_features).as_list()
                    for instance in cluster_data
                ]
            elif method == 'rankshap':
                explanations[str(cluster_id)] = rankshap(model, cluster_data, num_features) if model else None
            elif method == 'kernelshap':
                if model:
                    explanations[str(cluster_id)] = kernelshap(model, cluster_data, num_features)
                else:
                    # Use a dummy model for unsupervised tasks
                    def dummy_predict(data):
                        n_samples = data.shape[0]
                        return np.hstack([np.ones((n_samples, 1)) * 0.5, np.ones((n_samples, 1)) * 0.5])

                    explanations[str(cluster_id)] = kernelshap(dummy_predict, cluster_data, num_features)
            else:
                raise ValueError("Unsupported explanation method")

        # Save explanations with method-specific filename
        explanations_output_path = f"./Results/explanations_{method}.json"
        with open(explanations_output_path, 'w') as f:
            json.dump(explanations, f, indent=4)
        print(f"Explanations saved to: {explanations_output_path}")

        return explanations


# Load data
X_train, y_train, X_test, y_test, mapping_dict = load_data("./Data", "unsupervised_dataset")

# Initialize clustering and explanation pipeline with minimum 6 clusters and maximum 10 clusters
pipeline = ClusteringAndExplanation(clustering_type='kmeans', org=6, lim=10, stp=1)

# Perform clustering
cluster_labels = pipeline.perform_clustering(X_train)

# Visualize clusters
pipeline.visualize_clusters(X_train, cluster_labels)

# Train model (optional for some explanation methods)
model = None  # Use None for unsupervised tasks without a model
if y_train is not None:  # Train a model only if labels are provided
    model = train_model(X_train, y_train, "nn")

# Explain clusters
explanations = pipeline.explain_clusters(X_train, cluster_labels, method='lime', model=model)

# Output explanations
print("Pipeline completed successfully. Check the Results folder for outputs.")
