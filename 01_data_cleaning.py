"""Assignment 1: Data Cleaning using Python."""
import pandas as pd

# Example dataset intentionally includes a missing value and a duplicate.
df = pd.DataFrame({
    "Student": ["Asha", "Bharath", "Chandana", "Asha"],
    "Attendance": [92, 85, None, 92],
    "Marks": [88, 76, 81, 88]
})

print("Original dataset:\n", df)
print("\nMissing values per column:\n", df.isna().sum())

# Fill numeric missing values with the column median.
df["Attendance"] = df["Attendance"].fillna(df["Attendance"].median())

# Remove duplicate rows.
df = df.drop_duplicates().reset_index(drop=True)

print("\nCleaned dataset:\n", df)
print("\nMissing values after cleaning:\n", df.isna().sum())
df.to_csv("cleaned_dataset.csv", index=False)
print("\nSaved cleaned_dataset.csv")
