# Import the necessary packages:
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

np.random.seed(1234)

# Download the S&P 500 data from yfinance:
start = '2014-01-01'         
end = '2024-01-01'          
sp500 = yf.Ticker('^GSPC').history(start = start, end = end, interval='1d')['Close'] 
sp500 = sp500.reset_index() 
sp500['Date'] = sp500['Date'].dt.date
sp500.set_index('Date', inplace=True)
print(sp500.tail())

# Plot the data:
sp500.Close.plot(figsize = (15,7), color='blue', label='Closin Price')
plt.title('Closing prices S&P 500 (2014-2024)', size=24, pad=15)
plt.yticks(size=15); plt.xticks(size=15); plt.xlim(sp500.index.min(), sp500.index.max())
plt.ylabel('Price $', size=20, labelpad=13); plt.xlabel('Date', size=20, labelpad=13)
plt.show()

# Create a white noise time series and plot it:
sp500['wn'] = np.random.normal(loc=sp500.Close.mean(), scale=sp500.Close.std(), size=len(sp500))

sp500.wn.plot(figsize = (15,7), color='blue', linewidth=.7, label = 'White Noise')
plt.title('White Noise simulation', size=24, pad=15)
plt.yticks(size=15); plt.xticks(size=15); plt.xlim(sp500.index.min(), sp500.index.max())
plt.ylabel('Price $', size=20, labelpad=13); plt.xlabel('Date', size=20, labelpad=13)
plt.show()

# Compare the Close prices with the White Noise:
print(sp500.describe())

# Generate a Random Walk:
def rw_gen(T = 1, N = 100, mu = 0.1, sigma = 0.01, S0 = 20):
    dt = float(T)/N
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size = N)
    W = np.cumsum(W)*np.sqrt(dt)
    X = (mu-0.5*sigma**2)*t + sigma*W
    S = S0*np.exp(X)
    return S

dates = pd.date_range('2014-01-01', '2024-01-01')
T = (dates.max()-dates.min()).days / 365
N = dates.size
start_price = sp500.Close[0]
rw = pd.Series(rw_gen(T, N, mu=0.07, sigma=0.1, S0=start_price), index=dates)

sp500['rw'] = rw

sp500.rw.plot(figsize = (15,7), color='blue', label = 'Random Walk')
plt.title('Random Walk simulation', size=24, pad=15)
plt.yticks(size=15); plt.xticks(size=15); plt.xlim(sp500.index.min(), sp500.index.max())
plt.ylabel('Price $', size=20, labelpad=13); plt.xlabel('Date', size=20, labelpad=13)
plt.show()



# Plot the 3 series together:
fig, ax = plt.subplots(3, 1, figsize=(12, 9))

# Subplot #1:
ax[0].plot(sp500.index, sp500.Close, color='blue')
ax[0].set_title('Closing prices S&P 500 (2014-2024)', size=21, pad=10)
ax[0].set_xlabel('Date', size=16)
ax[0].set_ylabel('Price $', size=16)
ax[0].tick_params(axis='both', which='major', labelsize=14)

# Subplot #2:
ax[1].plot(sp500.index, sp500.wn, color='blue', linewidth=0.7)
ax[1].set_title('White Noise simulation', size=21, pad=10)
ax[1].set_xlabel('Date', size=16)
ax[1].set_ylabel('Price $', size=16)
ax[1].tick_params(axis='both', which='major', labelsize=14)

# Subplot #3:
ax[2].plot(sp500.index, sp500.rw, color='blue')
ax[2].set_title('Random Walk simulation', size=21, pad=10)
ax[2].set_xlabel('Date', size=16)
ax[2].set_ylabel('Price $', size=16)
ax[2].tick_params(axis='both', which='major', labelsize=14)

for ax in ax:
    ax.set_xlim(sp500.index.min(), sp500.index.max())
    ax.set_ylim(-500, 6000)
plt.tight_layout()
plt.subplots_adjust(hspace=0.5)
plt.show()


# Create a function to print the adf test in a more readable fashion:
def adf_test(column):
    adf_result = sts.adfuller(column)
    
    adf_output = pd.Series(adf_result[0:4], index=['Test Statistic','P-value','Lags Used','Number of Observations Used'])
    for key, value in adf_result[4].items():
        adf_output[f'Critical Value ({key})'] = value
    return adf_output
    
# Execute the Augmented Dickey-Fuller Test (ADF) for each series:
def adf_table(data, columns = []):
    table = pd.DataFrame()
    for i in range(0,len(columns)):
        adf = adf_test(data[str(columns[i])])
        table[str(columns[i])] = adf
        table
    table = table.round(decimals=4)
    return table
table = adf_table(sp500, columns=['Close', 'wn', 'rw'])
print(table)

# Time-series seasonal decomposition:
result = seasonal_decompose(sp500['Close'], model='additive', period=252)
result.plot()
plt.show()

# Extract the series without the trend:
close_no_trend = result.observed - result.trend
close_no_trend.isna().sum()
close_no_trend.fillna(0, inplace=True)

# Plot the differenced series:
close_no_trend.plot(figsize = (15,7), color='blue')
plt.title('Closing Prices without trend', size=24, pad=15)
plt.yticks(size=15); plt.xticks(size=15); plt.xlim(sp500.index.min(), sp500.index.max())
plt.ylabel('Price $', size=20, labelpad=13); plt.xlabel('Date', size=20, labelpad=13)
plt.show()

sp500['close_no_trend'] = close_no_trend

table2 = adf_table(sp500, columns=['Close', 'rw', 'wn', 'close_no_trend'])
print(table2)



