# Unsupervised_Clustering_
Explainability for Unsupervised Learning: Adapting SLIME, RankShap, and KernelShap to Clustering Tasks
## Overview
This project explores the adaptation of explainability techniques, including SLIME, RankShap, and KernelShap, to unsupervised learning tasks such as clustering. Traditionally designed for supervised learning, these methods are applied to clusters created by algorithms like K-Means to understand feature importance.

The pipeline integrates clustering and explainability, allowing users to:
- Perform clustering on unsupervised data.
- Generate explanations for clusters using SLIME, RankShap, or KernelShap.
- Evaluate clustering quality using silhouette scores.
- Save clustering labels and feature explanations for further analysis.

This project demonstrates the feasibility and challenges of explainable AI for unsupervised learning.
## Features
- **Clustering with Explainability**: Combines clustering algorithms like K-Means with explainability methods (SLIME, RankShap, KernelShap).
- **Customizable Clustering**: Supports various clustering types (`kmeans`, `hdbscan`, etc.).
- **Feature Explanations**: Explains feature importance for clusters using:
  - SLIME: Local Interpretable Model-Agnostic Explanations.
  - RankShap: Stable feature importance rankings.
  - KernelShap: Shapley values for feature contributions.
- **Output**:
  - Clustering labels (`clustering_labels.csv`).
  - Feature explanations for clusters (`explanations.json`).
- **Evaluation Metrics**:
  - Silhouette scores for assessing clustering quality.
