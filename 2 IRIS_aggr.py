import pandas as pd
df = pd.read_table(r"C:\Users\Gebruiker\Desktop\Data\iris.csv", sep=",")
df = df.groupby(['variety','Country'])['sepal.length'].agg('sum').reset_index() # group by sum

df['Country'] = df['Country'].replace('CN','China') # replace

print(df)
