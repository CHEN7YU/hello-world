import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_table(r"C:\Users\Gebruiker\Desktop\Data\Weather.csv", sep=",")
df.shape

# print(df[df['Date']== "1942/7/7"])

print(df.describe() )

# plot data
# df.plot(x='MinTemp', y='MaxTemp', style='o') 

# plot a column
plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(df['MaxTemp'])

df_x = df['MinTemp'].values.reshape(-1,1)
df_y = df['MaxTemp'].values.reshape(-1,1)

print(df_x.shape)
print(df_y.shape)

