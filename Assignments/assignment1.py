import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df_data = pd.read_csv(f"C:/Users/Lenovo/Downloads/raw_employee_data_one_row.csv",header=None)
columns=["Name","ID","Age","Gender","Department","Salary","Address"]
data_list=df_data.values.flatten().tolist()
records=[data_list[i:i+7] for i in range(0,len(data_list),7)]
df_clean=pd.DataFrame(records,columns=columns)

# 1st Qustion's Answer
deparment_count=df_clean['Department'].value_counts()
print("Number of employees in each department:",deparment_count)

#2nd Qustion's Answer

avg_salary=df_clean.groupby("Department")['Salary'].mean()
print("Average salary by group :",avg_salary)

# 3rd Qustion's Answer

engineersbw30to40 = df_clean[(df_clean['Age'].between(30,40))&(df_clean["Department"]=="Engineering")]
print(engineersbw30to40)

# 4th Qustion's Answer

std=df_clean["Salary"].std()
print(std)


#5th Qustion's Answer

mean_salary=df_clean["Salary"].mean()
above_mean=df_clean[df_clean["Salary"]>mean_salary]
count_above_mean=above_mean.shape[0]
print(count_above_mean)


# 6th Qustion's Answer

unique_ages = df_clean['Age'].dropna().unique()
sorted_ages=np.sort(unique_ages)
print(sorted_ages)

# 7th Qustion's Answer

department_counts = df_clean['Department'].value_counts()
plt.figure(figsize=(8, 6))
department_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Number of Employees in Each Department')
plt.xlabel('Department')
plt.ylabel('Number of Employees')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# 8th qustions Answer
age=df_clean["Age"]
plt.figure(figsize=(8, 6))
plt.hist(age, bins=8, color='yellow', edgecolor='black')
plt.title('Histogram of Employee Ages')
plt.xlabel('Age')
plt.ylabel('Number of Employees')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# 9th Qustion's Answer


gender_counts = df_clean['Gender'].value_counts()
plt.figure(figsize=(6, 6))
gender_counts.plot(kind='pie', autopct='%1.1f%%',  startangle=90, colors=['lightblue', 'lightpink'], 
                   title='Gender Distribution')
plt.show()

# 10th Qustion's Answer

df_scatter = df_clean.dropna(subset=['Age', 'Salary'])
plt.figure(figsize=(8, 6))
plt.scatter(df_scatter['Age'], df_scatter['Salary'], color='purple', alpha=0.6)
plt.title('Scatter Plot of Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.grid(True)
plt.tight_layout()

plt.show()