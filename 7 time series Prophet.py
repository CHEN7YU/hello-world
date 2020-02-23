# https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet
from matplotlib import pyplot as plt
from matplotlib.dates import MonthLocator, num2date
from matplotlib.ticker import FuncFormatter

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

# group by Order Date
furniture = furniture.groupby('Order Date')['Sales'].agg('sum').reset_index() # group by sum
# print(df)

# get 1st months of Order Date
furniture['month'] = furniture['Order Date'].dt.floor('d') - pd.offsets.MonthBegin(1)

# sum of monthly sales 
furniture = furniture.groupby('month')['Sales'].agg('sum').reset_index() 

# rename columns header
furniture.columns =["Order Date", "Sales"]
# print(furniture)

# plot data - descriptive
furniture["Sales"].plot( figsize=(15, 6))
plt.show()
# furniture.plot()

furniture.to_csv(r'C:\Users\Gebruiker\Desktop\Data\Output.csv', sep=',', index=False)

# Start Prophet section
# rename header
furniture = furniture.rename(columns={'Order Date': 'ds', 'Sales': 'y'})
furniture_model = Prophet(interval_width=0.95)
furniture_model.fit(furniture)

furniture_forecast = furniture_model.make_future_dataframe(periods=12, freq='M')
furniture_forecast = furniture_model.predict(furniture_forecast)


plt.figure(figsize=(18, 6))
furniture_model.plot(furniture_forecast, xlabel = 'Date', ylabel = 'Sales')
plt.title('Furniture Sales');


furniture_forecast_1 = furniture_forecast.loc[furniture_forecast['ds'] == "2018/1/31"]
print(furniture_forecast_1)

furniture_forecast.to_csv(r'C:\Users\Gebruiker\Desktop\Data\Output_f.csv', sep=',', index=False)
