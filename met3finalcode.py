import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import skfuzzy as fuzz
from minisom import MiniSom
import matplotlib.pyplot as plt
# Set seed for reproducibility
np.random.seed(2)
# Define the dimensions of your data matrix
N = 10  # Number of rows
M = 2   # Number of columns
# Create a random data matrix with some missing values
data_matrix = np.random.rand(N, M)
# Introduce missing values (replace 20% of values with NaN)
missing_percentage = 0.2
missing_mask = np.random.rand(N, M) < missing_percentage
data_matrix[missing_mask] = np.nan
# Display the random data matrix with missing values
print("Original Data Matrix:")
print(data_matrix)
# Define the replace_missing_values function
# Initialize new_data with mean, min, and max
def replace_missing_values(df):
    for col in range(df.shape[1]):
        if np.isnan(df[:, col]).any():
            col_mean = np.nanmean(df[:, col])
            df[:, col][np.isnan(df[:, col])] = col_mean
    return df
# Replace missing values
data_matrix_filled = replace_missing_values(data_matrix.copy())
def knn_imputation(data, k_neighbors=5):
    imputer = KNNImputer(n_neighbors=k_neighbors)
    return imputer.fit_transform(data)
# Impute missing values using KNN
data_matrix_imputed = knn_imputation(data_matrix_filled)
# Display the random data matrix with missing values
print("Data Matrix After Imputation:")
print(data_matrix_imputed)
# Define the KMeans clustering function
def kmeans_clustering(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(data)
# Perform KMeans clustering
kmeans_labels = kmeans_clustering(data_matrix_imputed)
# Define the Fuzzy C-Means clustering function
def fuzzy_c_means_clustering(data, n_clusters=3, m=2, max_iter=100, error=1e-6):
    cntr, u, _, _, _, _, _ = fuzz.cmeans(data.T, n_clusters, m, error, max_iter)
    return np.argmax(u, axis=0)
# Perform Fuzzy C-Means clustering
fcm_labels = fuzzy_c_means_clustering(data_matrix_imputed)
# Define the plot_clusters function
def plot_clusters(data, labels, title):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis')
    plt.title(title)
    plt.show()
# Plot the clusters
plot_clusters(data_matrix_imputed, kmeans_labels, "KMeans Clustering")
plot_clusters(data_matrix_imputed, fcm_labels, "Fuzzy C-Means Clustering")
