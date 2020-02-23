import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set(style="ticks")

dataset = pd.read_table("iris.csv", sep=',')

sns.pairplot(dataset, hue="variety")
plt.show()

# https://seaborn.pydata.org/examples/scatterplot_matrix.html
# https://seaborn.pydata.org/generated/seaborn.pairplot.html