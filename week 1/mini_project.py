import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Task 1
df_data=pd.read_csv("C:/Users/Lenovo/Downloads/heart.csv",)
# print(df_data)

# Task 2

#Average of patient Age
avg_age=df_data['age'].mean()
print("Average of patient age is :",avg_age)
#Average of patient BPs
avg_bp=df_data['trestbps'].mean()
print("Average of patient's bps is:",avg_bp)
# Average of Serum cholesterol (mg/dl) of patient
avg_chol=df_data['chol'].mean()
print("Average of Serum cholesterol (mg/dl) of patient:",avg_chol)
# Now median of these data 
age_median=df_data["age"].median()
bp_median=df_data["trestbps"].median()
choleserol_median=df_data["chol"].median()
print("median of patients Age,BPs and cholesrol is :",age_median,bp_median,choleserol_median)
# Now mode of these data 
age_mode=df_data['age'].mode()
print(age_mode)
# Now standard deviation of these data
age_std=df_data["age"].std()
print(age_std)




# Task 3 

# historgrams

plt.hist(df_data["age"],bins=10,rwidth=0.8,label='Age distribution')
plt.xlabel="Age of patients"
plt.ylabel="No of patients"
plt.legend()
plt.show()

# Boxplots

plt.figure(figsize=(6, 2))
df_data.boxplot(column=["age"])
plt.title("Boxplot of Age")
# plt.ylabel("Value")
plt.grid(True)
plt.show()

# "fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"

# Task 4

# corelation heatmaps

corr_matrix = df_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Heart Disease Dataset")
plt.show()