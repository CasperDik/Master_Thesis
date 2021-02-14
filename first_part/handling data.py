import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# import from excel
df = pd.read_excel(r'C:\Users\Casper Dik\OneDrive\Documenten\MSc Finance\Master Thesis\Data\Carbon Prices\ca'
                             r'rbon_prices_full.xlsx')
# rename column
df.rename(columns={"time": "Date", "CFI2Zc1": "Carbon_price"}, inplace=True)

# invert date and prices
df.Date = df.Date.values[::-1]
df.Carbon_price = df.Carbon_price.values[::-1]

# check for missing values
print(df.isnull().sum())

# describe data
print("descriptive stats carbon price full: \n", df["Carbon_price"].describe())
print("mean from panda: ", df.Carbon_price.mean())

# create return series
df["daily_returns"] = df.Carbon_price.pct_change()
print("descriptive stats daily returns: \n", df.daily_returns.describe())

# new dataframe from 2009
cp_2 = df.iloc[350:]
print("descriptive stats carbon price partially: \n", cp_2["Carbon_price"].describe())

# plot prices and returns
fig, axs = plt.subplots(2)
fig.suptitle('Carbon prices and returns')
axs[0].plot(cp_2.Date, cp_2.Carbon_price)
axs[1].plot(cp_2.Date, cp_2.daily_returns)
plt.show()
