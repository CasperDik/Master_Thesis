import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# import from excel
df = pd.read_excel(r'C:\Users\Casper Dik\OneDrive\Documenten\MSc Finance\Master Thesis\Data\Carbon Prices\ca'
                             r'rbon_prices_full.xlsx')
# rename column
df.rename(columns={"time": "Date", "CFI2Zc1": "price"}, inplace=True)

# invert date and prices
df.Date = df.Date.values[::-1]
df.price = df.price.values[::-1]

# check for missing values
print(df.isnull().sum())

# describe data
print("descriptive stats carbon price full: \n", df["price"].describe())
print("mean from panda: ", df.price.mean())

# create return series
df["daily_returns"] = df.price.pct_change()
df["log_returns"] = np.log(df.price) - np.log(df.price.shift(1))
print(df.head())

# new dataframe from 2008ish
cp_2 = df.iloc[350:]

# plot prices and returns
fig, axs = plt.subplots(3)
axs[0].plot(cp_2.Date, cp_2.price)
axs[1].plot(cp_2.Date, cp_2.daily_returns)
axs[2].plot(cp_2.Date, cp_2.log_returns)

axs[0].title.set_text("Price:")
axs[1].title.set_text("Returns:")
axs[2].title.set_text("Log returns:")

axs[0].get_xaxis().set_visible(False)
axs[1].get_xaxis().set_visible(False)

plt.show()
