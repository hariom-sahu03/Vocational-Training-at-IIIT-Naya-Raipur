import pandas as pd
df_row=pd.read_csv("D:/Programs/synthetic_students_one_row.csv",header=None)
columns=["Name","Roll no","Age","Gender","Address"]
data_list=df_row.values.flatten().tolist()
records=[data_list[i:i+5] for i in range(0,len(data_list),5)]
df_clean=pd.DataFrame(records,columns=columns)
print(df_clean.head())
df_clean.to_csv('cleaned_data.csv',index=False)
# for i in range(0,len(df_clean)):
#     if "Gender"=="Male":

male= df_clean[df_clean['Gender']=='Male'] 
print(male)      
