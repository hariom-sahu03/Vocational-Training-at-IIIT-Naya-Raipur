import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df_data=pd.read_excel("C:/Users/Lenovo/Downloads/train.csv.xlsx")
# print(df_data)

# Task 1
plt.bar(df_data['Survived'],df_data['Pclass'],label="passenger survived by class",color="blue")
plt.xlabel("survied ")
plt.ylabel("class")
# plt.show()

# Task 2
plt.scatter(df_data['Fare'],df_data['Age'],label="fare vs age",color="black",)
plt.xlabel("FARE")
plt.ylabel("Age")
# plt.show()


#Task 3
embark_counts = df_data['Embarked'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(embark_counts, labels=embark_counts.index, autopct='%1.1f%%', colors=['lightblue', 'lightgreen', 'lightcoral'])
plt.title("Passenger Count by Embarkation Port")
plt.show()
