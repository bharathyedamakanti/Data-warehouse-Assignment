"""Assignment 2: Data Reduction using feature selection and PCA."""
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

iris = load_iris(as_frame=True)
X = iris.data
print("Original shape:", X.shape)
print("\nCorrelation matrix:\n", X.corr().round(2))

# Reduce four numeric features to two principal components.
X_scaled = StandardScaler().fit_transform(X)
X_reduced = PCA(n_components=2).fit_transform(X_scaled)
reduced = pd.DataFrame(X_reduced, columns=["PC1", "PC2"])
reduced["species"] = [iris.target_names[i] for i in iris.target]
print("\nReduced dataset (first 5 rows):\n", reduced.head())
print("Reduced shape:", reduced.shape)
reduced.to_csv("iris_reduced_pca.csv", index=False)
print("\nSaved iris_reduced_pca.csv")
