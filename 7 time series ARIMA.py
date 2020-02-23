import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet

df = pd.read_excel(r"C:\Users\Gebruiker\Desktop\Data\Sample - Superstore.xls", sheet_name= "Orders")
furniture = df.loc[df['Category'] == 'Furniture']  # subset data 
# print(furniture['Order Date'].min(), furniture['Order Date'].max())  

print(furniture.shape) # data frame shape 

# drop columns
cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']
furniture.drop(cols, axis=1, inplace=True)  # drop columns
furniture = furniture.sort_values('Order Date')
furniture.isnull().sum()
# print(furniture)

# group by
df = furniture.groupby('Order Date')['Sales'].agg('sum').reset_index() # group by sum
# print(df)

furniture = df.set_index('Order Date')

# resample every 1 day of the month
y = furniture['Sales'].resample('MS').mean()

y.plot(figsize=(15, 6))
plt.show()

