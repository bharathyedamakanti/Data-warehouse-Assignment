"""Assignment 6: Demonstrate regression on a dataset."""
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = load_diabetes(as_frame=True)
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Model: Linear Regression")
print("Mean Absolute Error:", round(mean_absolute_error(y_test, predictions), 3))
print("Mean Squared Error:", round(mean_squared_error(y_test, predictions), 3))
print("R-squared:", round(r2_score(y_test, predictions), 3))

results = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
results.to_csv("regression_predictions.csv", index=False)
print("\nSample predictions:\n", results.head())
print("\nSaved regression_predictions.csv")
