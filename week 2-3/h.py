

import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score,confusion_matrix, accuracy_score, mean_absolute_error, mean_squared_error, r2_score,precision_score, recall_score,classification_report
data=pd.read_csv(r"C:\Users\Lenovo\Downloads\student_dataset.csv")
data.info()

data['Pass'] = (data['Final_Score'] >= 40).astype(int)
data
data[data.Final_Score<40]
x=data.drop(columns= ['Final_Score','Pass'],axis=1)
y=data['Pass']
data['Gender'] = data['Gender'].replace({'Male': 1, 'Female': 0})
data['Part_Time_Job'] = data['Part_Time_Job'].replace({'Yes': 1, 'No': 0})
print (data)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
log = LogisticRegression(max_iter=1000)
log.fit(x_train, y_train)

# Predict
pred = log.predict(x_test)

# Evaluate
print("📊 Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall:", recall_score(y_test, pred))
