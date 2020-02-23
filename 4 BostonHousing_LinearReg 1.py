import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split


df = pd.read_table(r"C:\Users\Gebruiker\Desktop\Data\boston_house_prices_qc.csv", sep=",")
df.dropna(how='all') # drop all nan columns

droprows = df[ df['GENDER'] == "UNKNOWN"].index # drop rows with condition
df.drop(droprows , inplace=True)

droprows = df[ pd.isna(df['GENDER']) ].index # drop rows with condition
df.drop(droprows , inplace=True)

df_x = df
df_x = df_x.drop(columns = ['MEDV'])
df_x = df_x.drop(columns = ['GENDER'])
print(df_x.describe())

df_y = df['MEDV']
print(df_y)

reg = linear_model.LinearRegression()

# split
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state=42)
reg.fit(x_train,y_train)

# predict
pred_results = reg.predict(x_test)
MSE1 = np.mean(pred_results - y_test)**2 # mean square error (MSE) < 1 -> good prediction
print(pred_results[5])


print(MSE1)

reg.score(x_test, y_test)  # model score