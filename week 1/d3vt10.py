import pandas as pd
df=pd.read_csv("D:\Programs\raw_employee_data_one_row.csv",header=None)
print(df)
columns=["Name","ID","Age","Gender","Department","Salary","Address"]
data_list=df.values.flatten().tolist()
records=[data_list[i:i+5] for i in range(0,len(data_list),5)]
df_clean=pd.DataFrame(records,columns=columns)
print(df_clean.head())
df_clean.to_csv('cleaned_data.csv',index=False)
