import pandas as pd

# Load the dataset directly from the URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')

# Display basic info
print("Initial Shape:", df.shape)
print(df.info())

# Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Drop duplicates if any
initial_shape = df.shape
df.drop_duplicates(inplace=True)
print(f"\nDuplicates removed: {initial_shape[0] - df.shape[0]}")

# Optional: handle outliers using IQR (Interquartile Range)
def remove_outliers_iqr(dataframe):
    Q1 = dataframe.quantile(0.25)
    Q3 = dataframe.quantile(0.75)
    IQR = Q3 - Q1
    return dataframe[~((dataframe < (Q1 - 1.5 * IQR)) | (dataframe > (Q3 + 1.5 * IQR))).any(axis=1)]

# Uncomment below if you want to remove outliers
# df = remove_outliers_iqr(df)
# print("Shape after outlier removal:", df.shape)

# Final summary
print("\nCleaned Data Preview:\n", df.head())
print("\nFinal Shape:", df.shape)





import pandas as pd

# Load the dataset from the URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')

# Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Display column names
print("Column Names:")
print(df.columns.tolist())

# Display first few rows as a table
print("\nFirst 5 rows of the dataset:")
print(df.head().to_string(index=False))




















from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Map target to names for clarity
df['target_name'] = df['target'].map({0: 'malignant', 1: 'benign'})

# Display class distribution
print("Class Distribution:")
print(df['target_name'].value_counts())

# Optional: percentage breakdown
print("\nPercentage Distribution:")
print(df['target_name'].value_counts(normalize=True) * 100)




































from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train KNN classifier (e.g., k=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Predict on test data
y_pred = knn.predict(X_test_scaled)

# Evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=data.target_names))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


from sklearn.metrics import precision_score

# Binary precision (default is for class '1' → benign)
print("Precision (benign):", precision_score(y_test, y_pred))

# Specify class label if you want precision for class '0' (malignant)
print("Precision (malignant):", precision_score(y_test, y_pred, pos_label=0))

