import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_table("iris.csv", sep=',')
print(dataset.describe())
#print(dataset.head())
#print(dataset['sepal.length'].head())

N=150
x=dataset['sepal.length']
y=dataset['sepal.width']
area=dataset['petal.width'] ** 7
colors = np.random.rand(N)

plt.scatter(x,y,  s=area, c=colors, alpha=0.5)
plt.xlabel('sepal.length', fontsize='large')
plt.ylabel('sepal.width', fontsize='large');

plt.show()
print(colors)

# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html#matplotlib.pyplot.scatter
# https://matplotlib.org/gallery/shapes_and_collections/scatter.html#sphx-glr-gallery-shapes-and-collections-scatter-py
