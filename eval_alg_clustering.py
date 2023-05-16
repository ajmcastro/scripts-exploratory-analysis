from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV data
data = pd.read_csv('path_to_your_file/data.csv')

# Display the first few rows of the DataFrame
print(data.head())

# Get basic statistics for each column (like count, mean, std, min, 25%, 50%, 75%, max)
print(data.describe(include='all'))

# Get information about the data types,columns, null value counts, memory usage etc
print(data.info())

# Assuming all columns are features
X = data

# Standardize features by removing the mean and scaling to unit variance
sc = StandardScaler()
X = sc.fit_transform(X)

# Reduce dimension to 2 with PCA for a 2D visualization
pca = PCA(n_components=2)
X = pca.fit_transform(X)

# Create a dictionary of different clustering algorithms we would like to test
clustering_algorithms = {
    "KMeans": KMeans(n_clusters=3),
    "DBSCAN": DBSCAN(eps=0.3),
    "AgglomerativeClustering": AgglomerativeClustering(n_clusters=3)
}

# Apply the clustering algorithms
for name, algorithm in clustering_algorithms.items():
    # Predict the clusters
    clusters = algorithm.fit_predict(X)

    # Plot the clusters
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
    plt.title(name)
    plt.show()

