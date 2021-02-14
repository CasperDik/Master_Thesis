import pandas as pd
import matplotlib.pyplot as plt

carbon_price = pd.read_excel(r'C:\Users\Casper Dik\OneDrive\Documenten\MSc Finance\Master Thesis\Data\Carbon Prices\ca'
                             r'rbon_prices_full.xlsx')
carbon_price.rename(columns={"time": "Date", "CFI2Zc1": "Carbon_price"}, inplace=True)
carbon_price.Date = carbon_price.Date.values[::-1]
carbon_price.Carbon_price = carbon_price.Carbon_price.values[::-1]

plt.plot(carbon_price.Date, carbon_price.Carbon_price)
# plt.show()

x = carbon_price.Carbon_price.to_numpy()
print(x[:5])
