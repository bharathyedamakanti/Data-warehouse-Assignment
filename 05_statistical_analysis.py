"""Assignment 5: Statistical analysis of a dataset."""
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.data
print("First five rows:\n", df.head())
print("\nDataset dimensions:", df.shape)
print("\nDescriptive statistics:\n", df.describe().round(3))
print("\nMean:\n", df.mean().round(3))
print("\nMedian:\n", df.median().round(3))
print("\nMode:\n", df.mode().iloc[0].round(3))
print("\nVariance:\n", df.var().round(3))
print("\nStandard deviation:\n", df.std().round(3))
print("\nCorrelation:\n", df.corr().round(3))
df.describe().to_csv("iris_statistical_summary.csv")
print("\nSaved iris_statistical_summary.csv")
