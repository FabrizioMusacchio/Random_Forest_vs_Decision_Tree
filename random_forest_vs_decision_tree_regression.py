"""
A simple script to compare the performance of a decision tree regressor and a random forest regressor.
"""
# %% IMPORTS
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
# %% REGRESSION PROBLEM

data = fetch_california_housing(as_frame=True)
X = data.data.loc[:, ['Longitude', 'Latitude', 'MedInc']]
y = data.target


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision tree regressor
dt_regressor = DecisionTreeRegressor()
dt_regressor.fit(X_train, y_train)

# Random forest regressor
rf_regressor = RandomForestRegressor(n_estimators=100)
rf_regressor.fit(X_train, y_train)

# Predict on the test set
dt_predictions = dt_regressor.predict(X_test)
rf_predictions = rf_regressor.predict(X_test)

# Calculate metrics:
dt_r2 = r2_score(y_test, dt_predictions)
rf_r2 = r2_score(y_test, rf_predictions)
dt_mse = mean_squared_error(y_test, dt_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)

# Plot predicted versus actual values
plt.figure(figsize=(10, 5))
plt.scatter(y_test, dt_predictions, color='b', label='Decision Tree' + f", R$^2$: {dt_r2:.3f}", alpha=0.4)
plt.scatter(y_test, rf_predictions, color='r', label='Random Forest' + f", R$^2$: {rf_r2:.3f}", alpha=0.4)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.show()

# Print metrics:
print("Decision Tree Mean Squared Error:", dt_mse)
print("Random Forest Mean Squared Error:", rf_mse)
print("Decision Tree R-squared:", dt_r2)
print("Random Forest R-squared:", rf_r2)
# %% END