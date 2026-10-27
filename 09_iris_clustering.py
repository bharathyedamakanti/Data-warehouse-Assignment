"""Assignment 9: Demonstration of simple K-means clustering on Iris."""
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, adjusted_rand_score

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target
X_scaled = StandardScaler().fit_transform(X)

# K-means is unsupervised: cluster IDs do not necessarily match species IDs.
model = KMeans(n_clusters=3, random_state=7, n_init=10)
cluster_ids = model.fit_predict(X_scaled)

print("Cluster distribution:")
print(pd.Series(cluster_ids).value_counts().sort_index())
print("\nAdjusted Rand Index against known species labels:",
      round(adjusted_rand_score(y, cluster_ids), 3))
print("\nCross-tabulation of species and cluster IDs:")
print(pd.crosstab(pd.Series([iris.target_names[i] for i in y], name="Species"),
                  pd.Series(cluster_ids, name="Cluster")))

results = X.copy()
results["cluster"] = cluster_ids
results["species"] = [iris.target_names[i] for i in y]
results.to_csv("iris_clustering_results.csv", index=False)
print("\nSaved iris_clustering_results.csv")
