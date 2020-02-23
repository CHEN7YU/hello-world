import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_table("iris.csv", sep=',')
print(dataset.describe())

V = dataset[dataset['variety'] == "Virginica"]
S = dataset[dataset['variety'] == "Setosa"]

X1 = V['sepal.length']
X2 = S['sepal.length']

Y1 = V['sepal.width']
Y2 = S['sepal.width']


plt.subplot(2, 1, 1)
plt.scatter(X1,Y1,   c="green", alpha=0.5)
plt.title('A tale of 2 subplots')
plt.ylabel('Virginica sepal.width')

plt.subplot(2, 1, 2)
plt.scatter(X2,Y2,   c="red", alpha=0.5)
plt.xlabel('sepal.length')
plt.ylabel('Setosa sepal.width')

plt.show()

# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html#matplotlib.pyplot.subplot
