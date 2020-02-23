import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

tips = sns.load_dataset("tips")
print(tips)
print(tips.describe())

g = sns.jointplot("total_bill", "tip", data=tips, kind="reg", color="r", height=7)
plt.show()


# https://seaborn.pydata.org/examples/regression_marginals.html
# https://seaborn.pydata.org/generated/seaborn.jointplot.html

