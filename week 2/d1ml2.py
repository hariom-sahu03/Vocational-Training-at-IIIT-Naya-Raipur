import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_absolute_error, mean_squared_error, r2_score
data= pd.read_csv("C:/Users/Lenovo/Downloads/Property_Dataset.csv")
data.head()
encoder = LabelEncoder()
data['Property_Type'] = encoder.fit_transform(data['Property_Type'])
data['Category'] = encoder.fit_transform(data['Category'])
