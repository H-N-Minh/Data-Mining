# KDDM1 Course Assignments

This repository showcases my solutions for the assignments in the KDDM1 course, focusing on practical data analysis and machine learning tasks.

### Technical Overview

*   **Programming Language:** Python
*   **Key Libraries:** Pandas, Scikit-learn, Matplotlib, LightGBM, Optuna
*   **Core Concepts Covered:** Exploratory Data Analysis (EDA), Statistical Correlation, Anomaly Detection, Data Imputation, Natural Language Processing (NLP), Unsupervised Learning (Clustering, PCA), and Supervised Learning (Classification).

---

## Assignment 1: Data Analysis Fundamentals (`A1` Folder)

This assignment covered core data wrangling and analysis techniques.

*   **1. Visual Data Analysis:** Explored a dataset by creating various charts to find hidden patterns and relationships between its features.

*   **2. Correlation:** Used statistical methods (like Pearson and Spearman) to identify and measure the strength of linear and non-linear relationships within a large, mostly random dataset.

*   **3. Outlier Detection:** Implemented algorithms like Local Outlier Factor (LOF) and DBSCAN to find two different kinds of anomalies: single, isolated data points and entire sparse regions.

*   **4. Missing Values:** Analyzed an incomplete dataset to understand why data was missing and used appropriate strategies (like regression and mode imputation) to fill in the gaps intelligently.

---

## Assignment 2: NLP and Classification Pipeline (`A2` Folder)

This assignment involved building an end-to-end machine learning pipeline for text analysis and prediction.

*   **1. Feature Engineering:** Processed raw text from European Parliament debates, converting the speeches into numerical data that represents the most important keywords and phrases (using TF-IDF).

*   **2. Clustering:** Used the K-Means algorithm to automatically group the debate speeches into thematic clusters. This successfully identified 18 distinct topics being discussed.

*   **3. Dimensionality Reduction:** Used PCA to simplify the complex text data, allowing the clusters and their relationships to be visualized on a 2D plot.

*   **4. Classification:** Built a LightGBM model to predict the outcome of a chess endgame. After optimizing its parameters, the model achieved a high accuracy of **97.6%**.
