import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load the data
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df['target_name'] = df['target'].map({0: 'malignant', 1: 'benign'})

# Choose two features to plot
x_feature = 'mean radius'
y_feature = 'mean texture'

# Create the scatter plot
plt.figure(figsize=(8, 6))
for label in df['target'].unique():
    subset = df[df['target'] == label]
    plt.scatter(subset[x_feature], subset[y_feature], label=data.target_names[label], alpha=0.7)

plt.xlabel(x_feature)
plt.ylabel(y_feature)
plt.title('Scatter Plot: {} vs {}'.format(x_feature, y_feature))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()











import pandas as pd
import matplotlib.pyplot as plt

# Load the uploaded CSV file
file_path = "/mnt/data/synthetic_salary_data_raw.csv"
df = pd.read_csv(file_path)

# Print column names to help choose features
print("Columns in dataset:")
print(df.columns)

# Example scatter plot (you can change the column names accordingly)
x_feature = 'YearsExperience'   # example column
y_feature = 'Salary'            # example column

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df[x_feature], df[y_feature], color='blue', alpha=0.6)
plt.xlabel(x_feature)
plt.ylabel(y_feature)
plt.title(f'Scatter Plot: {x_feature} vs {y_feature}')
plt.grid(True)
plt.tight_layout()
plt.show()





























import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the data
file_path = "/mnt/data/synthetic_salary_data_raw.csv"
df = pd.read_csv(file_path)

# Preview columns
print("Columns:", df.columns)
print(df.head())

# Define features (X) and target (y)
# Adjust these column names if your file has different names
X = df[['YearsExperience']]  # feature(s)
y = df['Salary']             # target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Plot actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Linear Regression: Salary vs Experience')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
































X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output model results
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R^2 Score:", r2)
