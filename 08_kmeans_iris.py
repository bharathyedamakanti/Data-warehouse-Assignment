"""Assignment 8: K-means algorithm for the Iris dataset."""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

iris = load_iris(as_frame=True)
X = iris.data
X_scaled = StandardScaler().fit_transform(X)

model = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = model.fit_predict(X_scaled)

print("Cluster counts:\n", pd.Series(clusters).value_counts().sort_index())
print("Silhouette score:", round(silhouette_score(X_scaled, clusters), 3))

pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)
plt.figure()
plt.scatter(coords[:, 0], coords[:, 1], c=clusters)
plt.title("K-means Clustering on Iris (PCA view)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.savefig("iris_kmeans_clusters.png", dpi=150)
print("Saved iris_kmeans_clusters.png")

output = X.copy()
output["cluster"] = clusters
output["actual_species"] = [iris.target_names[i] for i in iris.target]
output.to_csv("iris_kmeans_results.csv", index=False)
print("Saved iris_kmeans_results.csv")
