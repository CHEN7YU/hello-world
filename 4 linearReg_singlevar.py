import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_table(r"C:\Users\Gebruiker\Desktop\Data\Weather.csv", sep=",")
print(df.shape)

df = df[["MaxTemp","MinTemp"]]

# print(df[df['Date']== "1942/7/7"])

# print(df.describe() )

# plot data
df.plot(x='MinTemp', y='MaxTemp', style='o') 

# plot a column
plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(df['MaxTemp'])

df_x = df['MinTemp'].values.reshape(-1,1)
df_y = df['MaxTemp'].values.reshape(-1,1)


reg = linear_model.LinearRegression()

# split
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state=42)
reg.fit(x_train,y_train)

print("Coef / Slope ",reg.coef_)
print("intercept ", reg.intercept_)
# Y = m * X + b (m is coefficient and b is intercept)

# predict
pred_results = reg.predict(x_test)
print(pred_results.shape)

# r2 square score
print(r2_score(y_test, pred_results))  # model score r squared, R2=1 very close to the reg line

# add predict rows to the df
print("result shape ",pred_results.shape)
print("x_test shape ",x_test.shape)

print(type(x_test))
x_test = pd.DataFrame(x_test) # convert to pandas
x_test.columns = ['Min Temp']  # add column name
print(type(x_test))

x_test["pred results"] = pred_results
x_test["Max Temp"] = y_test

x_test.head()



plt.plot(x_test['Min Temp'], x_test["pred results"], color='g')
plt.plot(x_test['Min Temp'], x_test['Max Temp'] , color='orange')
plt.xlabel('Min Temp')
plt.ylabel('Max Temp')
plt.show()