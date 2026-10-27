"""Assignment 3: Data Transformation using Python."""
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler

iris = load_iris(as_frame=True)
df = iris.data.copy()
print("Original data:\n", df.head())

# Min-max normalization scales values to the 0–1 range.
normalized = pd.DataFrame(
    MinMaxScaler().fit_transform(df),
    columns=[f"{c}_normalized" for c in df.columns]
)

# Standardization gives each feature approximately mean 0 and standard deviation 1.
standardized = pd.DataFrame(
    StandardScaler().fit_transform(df),
    columns=[f"{c}_standardized" for c in df.columns]
)

print("\nMin-max normalized data:\n", normalized.head())
print("\nStandardized data:\n", standardized.head())
normalized.to_csv("iris_normalized.csv", index=False)
standardized.to_csv("iris_standardized.csv", index=False)
print("\nSaved iris_normalized.csv and iris_standardized.csv")
