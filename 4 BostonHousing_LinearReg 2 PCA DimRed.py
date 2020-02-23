import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

df = pd.read_table(r"C:\Users\Gebruiker\Anaconda3\Lib\site-packages\sklearn\datasets\data\boston_house_prices_qc.csv", sep=",")
# print(bt)
df_x = df
df_x = df_x.drop(columns = ['MEDV'])
df_x = df_x.drop(columns = ['GENDER'])

# PCA
pca = PCA(n_components = 3, whiten = 'True')
df_x_red = pca.fit(df_x).transform(df_x)
print(df_x_red)

df_y = df['MEDV']
print(df_y)

reg = linear_model.LinearRegression()

# split
x_train, x_test, y_train, y_test = train_test_split(df_x_red, df_y, test_size=0.33, random_state=42)
reg.fit(x_train,y_train)

# predict
pred_results = reg.predict(x_test)
# reg.predict_proba(x_test)  # this can predict % of a individual to buy insurance or not
MSE1 = np.mean(pred_results - y_test)**2 # mean square error (MSE) < 1 -> good prediction
print(MSE1)

reg.score(x_test, y_test)  # model score