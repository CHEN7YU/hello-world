import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_table(r"C:\Users\Gebruiker\Desktop\Data\boston_house_prices_qc.csv", sep=",")
df.dropna(how='all') # drop all nan columns

droprows = df[ df['GENDER'] == "UNKNOWN"].index # drop rows with condition
df.drop(droprows , inplace=True)

droprows = df[ pd.isna(df['GENDER']) ].index # drop rows with condition
df.drop(droprows , inplace=True)

df_x = df
df_x = df_x.drop(columns = ['MEDV'])
df_x = df_x.drop(columns = ['GENDER'])
# print(df_x.describe())

df_y = df['MEDV']
#print(df_y)

reg = linear_model.LinearRegression()

# split
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state=42)
reg.fit(x_train,y_train)

# predict
pred_results = reg.predict(x_test)
# print(pred_results[5])

# model score r squared, R2=1 very close to the reg line
print(r2_score(y_test, pred_results))  

x_test["pred results"] = pred_results
x_test["true results"] = y_test

x_test = x_test.head(4)
print(x_test)

