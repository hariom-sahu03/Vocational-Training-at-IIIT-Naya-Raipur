import numpy as np
import pandas as pd
np.random.seed(42)
true_values=np.random.normal(loc=50, scale=10, size=1000)
predicted_values=true_values+np.random.normal(loc=0, scale=5, size=1000)
true_values[::50]+=np.random.normal(loc=20, scale=10,size=20)

data=pd.DataFrame({'true_values':true_values, 'predicted_values':predicted_values})
from scipy.stats import zscore
import matplotlib.pyplot as plt
import numpy as np

def scatter_plot_with_outliers(data):
    plt.figure(figsize=(10, 6))  # Set figure size

    # Scatter plot for all data points
    plt.scatter(data['True'], data['Predicted'], alpha=0.7, label='Data Points')

    # Calculate Z-Scores for outlier detection
    z_scores = zscore(data[['True', 'Predicted']])
    outliers = np.abs(z_scores) > 3  # Identify outliers with Z-Score > 3

    # Highlight outliers in red
    plt.scatter(
        data['True'][outliers.any(axis=1)],
        data['Predicted'][outliers.any(axis=1)],
        color='red', label='Outliers'
    )

    # Add labels, legend, and grid
    plt.title("Scatter Plot with Outliers Marked")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid()
    plt.show()

# Example usage:
# scatter_plot_with_outliers(data)  # `data` should be a DataFrame with 'True' and 'Predicted' columns

import matplotlib.pyplot as plt

def box_plot_with_outliers(data):
    plt.figure(figsize=(12, 6))  # Set figure size

    # Plot for True Values
    plt.subplot(1, 2, 1)
    plt.boxplot(data['True'], patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.title("Box Plot of True Values")
    plt.ylabel("Values")

    # Plot for Predicted Values
    plt.subplot(1, 2, 2)
    plt.boxplot(data['Predicted'], patch_artist=True, boxprops=dict(facecolor='lightgreen'))
    plt.title("Box Plot of Predicted Values")

    plt.tight_layout()
    plt.show()

# Example usage:
# data = pd.DataFrame({'True': y_test, 'Predicted': y_pred})
# box_plot_with_outliers(data)

import matplotlib.pyplot as plt
import numpy as np

def detect_outliers_iqr(data):
    for col in data.columns:
        Q1 = data[col].quantile(0.25)  # First quartile
        Q3 = data[col].quantile(0.75)  # Third quartile
        IQR = Q3 - Q1  # Interquartile range

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        outliers = (data[col] < lower_bound) | (data[col] > upper_bound)

        print(f"\nOutliers in '{col}' using IQR:")
        print(data[outliers])

        # Visualize outliers in a box plot
        plt.figure(figsize=(10, 5))
        plt.boxplot(data[col], patch_artist=True, boxprops=dict(facecolor='lightblue'))
        plt.scatter(np.ones(len(data[outliers][col])), data[outliers][col], color='red', label='Outliers')

        plt.title(f"Box Plot of '{col}' with IQR Outliers Marked")
        plt.legend()
        plt.grid()
        plt.show()

# Example usage:
# data = pd.DataFrame({'True': y_test, 'Predicted': y_pred})
# detect_outliers_iqr(data)


from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

def detect_outliers_isolation_forest(data):
    clf = IsolationForest(contamination=0.05, random_state=42)  # Initialize model
    preds = clf.fit_predict(data)  # Predict inliers and outliers

    outliers = preds == -1  # -1 indicates outliers

    # Print outliers
    print("Outliers detected using Isolation Forest:")
    print(data[outliers])

    # Scatter plot highlighting outliers
    plt.figure(figsize=(10, 6))
    plt.scatter(data['True'], data['Predicted'], alpha=0.7, label='Inliers')
    plt.scatter(data['True'][outliers], data['Predicted'][outliers], color='red', label='Outliers')

    plt.title("Scatter Plot with Isolation Forest Outliers")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid()
    plt.show()

# Example usage:
# data = pd.DataFrame({'True': y_test, 'Predicted': y_pred})
# detect_outliers_isolation_forest(data)
