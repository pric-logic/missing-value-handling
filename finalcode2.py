import numpy as np
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import skfuzzy as fuzz
from minisom import MiniSom
import matplotlib.pyplot as plt
import pandas as pd

# Set seed for reproducibility
np.random.seed(2)

# Define the dimensions of your data matrix
N = 10 # Number of rows
M = 2   # Number of columns

# Create a random data matrix with some missing values
data_matrix = np.random.rand(N, M)

# Introduce missing values (replace 10% of values with NaN)
missing_percentage = 0.2
missing_mask = np.random.rand(N, M) < missing_percentage
data_matrix[missing_mask] = np.nan

# Display the random data matrix with missing values
print("Original Data Matrix:")
print(data_matrix)

# Define the replace_missing_values function
def replace_missing_values(data):
    N, M = data.shape
    for r in range(N):
        for c in range(M):
            if np.isnan(data[r, c]):
                data[r, c] = 0
    return data

# Replace missing values
data_matrix_filled = replace_missing_values(data_matrix)

def knn_imputation(data, k_neighbors=5):
    imputer = KNNImputer(n_neighbors=k_neighbors)
    return imputer.fit_transform(data)
# Impute missing values using KNN
#data_matrix_imputed = knn_imputation(data_matrix_filled)

# Display the random data matrix with missing values
print("Data Matrix:")
print(data_matrix_filled)

# Define the KMeans clustering function
def kmeans_clustering(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(data)

# Perform KMeans clustering
kmeans_labels = kmeans_clustering(data_matrix_filled)

# Define the Fuzzy C-Means clustering function
def fuzzy_c_means_clustering(data, n_clusters=3, m=2, max_iter=100, error=1e-6):
    cntr, u, _, _, _, _, _ = fuzz.cmeans(data.T, n_clusters, m, error, max_iter)
    return np.argmax(u, axis=0)

# Perform Fuzzy C-Means clustering
fcm_labels = fuzzy_c_means_clustering(data_matrix_filled)

'''# Define the SOM clustering function
def som_clustering(data_matrix):
    num_dimensions = 10
    map_size = 10
    learning_rate = 0.5
    som = MiniSom(map_size, map_size, num_dimensions, sigma=1.0, learning_rate=learning_rate)
    data = pd.DataFrame(data_matrix).to_numpy()
    max_iter = 1000
    for i in range(max_iter):
        som.train(data, 500, verbose=False)
    bmus = som.win_map(data)
    som_labels = []
    for i in range(data.shape[0]):
        som_labels.append(bmus[i])
    return som_labels

# Perform SOM clustering
som_labels = som_clustering(data_matrix_imputed)'''

# Define the plot_clusters function
def plot_clusters(data, labels, title):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis')
    plt.title(title)
    plt.show()

# Plot the clusters
plot_clusters(data_matrix_filled, kmeans_labels, "KMeans Clustering")
plot_clusters(data_matrix_filled, fcm_labels, "Fuzzy C-Means Clustering")
#plot_clusters(data_matrix_imputed, som_labels, "SOM Clustering")