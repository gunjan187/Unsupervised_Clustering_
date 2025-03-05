# Unsupervised_Clustering_
Explainability for Unsupervised Learning: Adapting SLIME and KernelShap to Clustering Tasks
## Overview

This repository presents a comprehensive approach to **unsupervised clustering**, integrating explainability techniques to enhance the interpretability of clustering results. Traditional clustering algorithms often function as **black-box models**, making it challenging to understand the rationale behind cluster assignments. This project addresses this issue by leveraging **SHAP-based explainability** to analyze and rank feature importance in unsupervised settings.  

Through a combination of **clustering techniques (e.g., K-Means)** and **explainability methods (e.g., KernelSHAP)**, this research provides insights into how different features contribute to clustering decisions. The approach is validated on multiple datasets to demonstrate its effectiveness in real-world scenarios.  

---

## **Key Features**  
✔ **Implementation of Unsupervised Clustering Algorithms**  
- K-Means clustering with evaluation using **Silhouette Score**  

✔ **Explainability Methods for Clustering**  
- Application of **KernelSHAP** to understand feature contributions  
- Comparative analysis of feature importance rankings across different clusters  

✔ **Benchmark Datasets**  
- **bank.csv** (Financial transactions data)  
- **brca_small.csv** (Breast cancer dataset)  
- **census_X.csv** (Demographic and income data)  
- **credit.csv** (Credit risk assessment dataset)  

✔ **Reproducible & Well-Structured Code**  
- Modular implementation using Python  
- Jupyter notebooks for experimentation and visualization  
- Well-documented scripts for clustering and interpretability  

---

## **Project Motivation**  
Unsupervised clustering is widely used in data analysis, anomaly detection, and customer segmentation. However, its lack of transparency makes decision-making difficult. This project aims to **bridge the gap between clustering and interpretability**, helping researchers and practitioners understand how and why data points are grouped together. By integrating SHAP-based explanations, we provide a method to extract meaningful insights from unsupervised models.  

---

## **Repository Structure**  
```
📂 Unsupervised_Clustering/
│── datasets/         # Contains the benchmark datasets used for clustering
│── notebooks/        # Jupyter notebooks for clustering & explainability analysis
│── src/             # Python scripts for clustering, feature ranking, and SHAP
│── results/         # Evaluation metrics, visualizations, and findings
│── requirements.txt  # List of dependencies
│── README.md        # Project documentation
```

---

## **Getting Started**  
### **Installation**  
1️⃣ Clone the repository:  
```bash
git clone https://github.com/gunjan187/Unsupervised_Clustering.git
cd Unsupervised_Clustering
```  
2️⃣ Install required dependencies:  
```bash
pip install -r requirements.txt
```  
3️⃣ Run the clustering and explainability analysis:  
```bash
python src/main.py
```

---

## **References & Acknowledgments**  
This project is inspired by the work of **[Jeremy Goldwasser](https://github.com/jeremy-goldwasser/feature-rankings)** on feature rankings in explainable AI. The datasets were sourced from publicly available repositories for research and benchmarking purposes.  
