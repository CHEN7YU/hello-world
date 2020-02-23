import pandas as pd

Customer = pd.read_table(r"C:\Users\Gebruiker\Desktop\Data\Customers.txt", sep="|")
Order = pd.read_table(r"C:\Users\Gebruiker\Desktop\Data\Orders.txt", sep="|")
Order_Details = pd.read_table(r"C:\Users\Gebruiker\Desktop\Data\Order_Details.txt", sep="|")

missing_value = Customer.isnull().values.any()  # check if any columns is null
missing_value_details = Customer.isnull().sum() # check sum of missing values
print(missing_value)
print(missing_value_details)

Customer["Fax"].fillna("No Fax", inplace = True)  # replace nan with string 

Order.to_csv(r'C:\Users\Gebruiker\Desktop\Data\Output.txt', sep='|', index=False)
# export to csv

Order['Freight'] = Order['Freight'].replace(',','.')

# print(Order['Freight'].describe())

V = Customer[(Customer['City'] == 'London' ) & (Customer['Address'].str.contains('King'))] # Subset select
# V = V[['City', 'Address']]  # Subset

#print(V)
Order_w_details = pd.merge(Order,Order_Details, on="OrderID", how="inner")
df = pd.merge(Order_w_details,Customer, on="CustomerID", how="inner")

print(df.describe())
#print(df.head(4))

Order.drop
Customer.drop