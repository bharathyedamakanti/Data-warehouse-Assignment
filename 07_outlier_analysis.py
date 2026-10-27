"""Assignment 7: Outlier analysis using the IQR method."""
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.data.copy()
outlier_rows = pd.Series(False, index=df.index)

for column in df.columns:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    column_outliers = (df[column] < lower) | (df[column] > upper)
    outlier_rows |= column_outliers
    print(f"{column}: lower={lower:.2f}, upper={upper:.2f}, "
          f"outliers={int(column_outliers.sum())}")

outliers = df[outlier_rows]
print("\nTotal rows flagged as outliers:", len(outliers))
print("\nOutlier rows:\n", outliers)
outliers.to_csv("iris_outliers.csv", index=False)
print("\nSaved iris_outliers.csv")
