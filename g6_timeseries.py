import pandas as pd
import matplotlib.pyplot as plt
# df = pd.read_csv('https://github.com/QuantEcon/QuantEcon.lectures.code/raw/master/pandas/data/test_pwt.csv')
dataset = pd.read_table("UNRATE.csv", sep=',')


print(dataset.head())
print(min(dataset['DATE']))
print(max(dataset['DATE']))

dataset.plot()
plt.show()


# https://lectures.quantecon.org/py/pandas.html
