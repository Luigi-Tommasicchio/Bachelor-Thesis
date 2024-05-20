# Import the relevant packages
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Download data from yahoo finance
ticker = 'AAPL'
start_date = '2019-01-01'
end_date = '2024-01-01'

apple = yf.Ticker(ticker).history(ticker, start=start_date, end=end_date, interval='1d')[['Open', 'Close', 'High', 'Low', 'Volume']]

apple.head()

# Check for datatype and NaNs
apple.info()
apple.isna().sum()

# Reformat the index from 'datetime' to 'date'
apple = apple.reset_index()
apple['Date'] = pd.to_datetime(apple['Date'].dt.date)
apple.set_index('Date', inplace=True)
apple.head()

apple.info()

# Plot the data with a Candlestick Chart
import mplfinance as mpf
fig, ax = mpf.plot(apple, type='candle',volume=True, style='yahoo', 
                ylabel='Prezzo',ylabel_lower='Volume', xlabel = 'Data', show_nontrading=False, 
                figratio=(4,3), figscale=1.2, xrotation=45, returnfig=True)
ax[0].set_title('Candlestick chart AAPL', fontsize=18, loc='center')
plt.show()

plt.rcParams.keys()

# Create a correlation matrix for all the features
corr = apple.select_dtypes('number').corr()

import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='Blues', fmt=".2f")
sns.set_theme(font_scale=1.5)
plt.xlabel(size=15)
plt.ylabel(size=15)
plt.grid(None)
plt.title('Heatmap delle Correlazioni', size=30, y=1.02)
plt.show()

# Extract just the closing prices' series
close = apple['Close']
close.head()
close.info()

plt.rcParams.update({'figure.figsize':(15,7),'xtick.labelsize': 16, 'ytick.labelsize': 16})
close.plot(figsize = (15,7), color='blue', label='Prezzo di Chiusura')
plt.title('Prezzi di Chiusura AAPL (2019-2024)', size=24, pad=15)
plt.yticks(size=15)
plt.xticks(size=15)
plt.xlim(close.index.min(), close.index.max())
plt.ylabel('Prezzo $', size=20, labelpad=13)
plt.xlabel('Data', size=20, labelpad=13)
plt.show()

# Split the data in train and test sets
split = int(len(close)*0.8)

close_train = close[:split]
close_test = close[split:]

close_train = close_train.asfreq('b')
close_train.isna().sum()
close_train = close_train.fillna(method = 'ffill')
close_train.isna().sum()

close_test = close_test.asfreq('b')
close_test.isna().sum()
close_test = close_test.fillna(method = 'ffill')
close_test.isna().sum()

close_train.tail(2)
close_test.head(2)

print('Dimensione set di Train: ' + str(len(close_train)) + 
      '\nDimensione set di Test: ' + str(len(close_test)))

plt.rcParams.update({'figure.figsize':(15,7),'xtick.labelsize': 19, 'ytick.labelsize': 18})
close_train.plot(label='Train set', color='blue', figsize=(15,7))
close_test.plot(label='Test set', color='red')
plt.yticks(size=15)
plt.xticks(size=15)
plt.xlim(close.index.min(), close.index.max())
plt.ylabel('Prezzo $', size=20, labelpad=13)
plt.xlabel('Data', size=20, labelpad=13)
plt.title('Prezzi di chiusura AAPL', size=24, pad=15)
plt.legend(fontsize=20)
plt.show()


# Create three benchmarks:
close_test = close_test.reset_index()
close_test = pd.DataFrame({'Close':close_test.Close,
                           'Date':close_test.Date})
close_test = close_test.set_index('Date')

# MEAN benchmark 
mean_forecast = close_train.mean().repeat(len(close_test))
close_test['mean_forecast'] = mean_forecast

# NAIVE benchmark 
naive_forecast = close_train[-1] 
close_test['naive_forecast'] = naive_forecast

# DRIFT benchmark 
drift = (close_train[-1] - close_train[0]) / len(close_train)
drift
drift_forecast = []
for i in range(len(close_test)):
    forecast = close_train[-1] + (drift * i)
    drift_forecast.append(forecast)

close_test['drift_forecast'] = drift_forecast

# Plot the benchmarks
import matplotlib.patheffects as pe
plt.rcParams.update({'figure.figsize':(15,7),'xtick.labelsize': 20, 'ytick.labelsize': 20})
close_train.plot(label='Train set', color='blue', figsize=(15,7))
close_test.Close.plot(label='Test set', color='red')
close_test.naive_forecast.plot(label='Naive Forecast', color='green', lw=2)
close_test.drift_forecast.plot(label='Drift Forecast', color='purple', lw=2)
close_test.mean_forecast.plot(label='Mean Forecast', color='gold', lw=2, path_effects=[pe.Stroke(linewidth=2.5, foreground='black'), pe.Normal()])
plt.ylabel('Prezzo $', size=20)
plt.xlabel('Data', size=20)
plt.yticks(size=18)
plt.xticks(size=18)
plt.legend(fontsize=17)
plt.show()

# Calculate several error measures for each benchmark method
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

mean_mse = round(mean_squared_error(close_test.Close, close_test.mean_forecast),2)
mean_rmse = round(root_mean_squared_error(close_test.Close, close_test.mean_forecast),2)
mean_mae = round(mean_absolute_error(close_test.Close, close_test.mean_forecast),2)
mean_mape = round(mean_absolute_percentage_error(close_test.Close, close_test.mean_forecast),2)

naive_mse = round(mean_squared_error(close_test.Close, close_test.naive_forecast),2)
naive_rmse = round(root_mean_squared_error(close_test.Close, close_test.naive_forecast),2)
naive_mae = round(mean_absolute_error(close_test.Close, close_test.naive_forecast),2)
naive_mape = round(mean_absolute_percentage_error(close_test.Close, close_test.naive_forecast),2)

drift_mse = round(mean_squared_error(close_test.Close, close_test.drift_forecast),2)
drift_rmse = round(root_mean_squared_error(close_test.Close, close_test.drift_forecast),2)
drift_mae = round(mean_absolute_error(close_test.Close, close_test.drift_forecast),2)
drift_mape = round(mean_absolute_percentage_error(close_test.Close, close_test.drift_forecast),2)

benchmarks_errors = pd.DataFrame({
    'MSE': [mean_mse, naive_mse, drift_mse],
    'RMSE': [mean_rmse, naive_rmse, drift_rmse],
    'MAE': [mean_mae, naive_mae, drift_mae],
    'MAPE': [mean_mape, naive_mape, drift_mape]},
    index = ['Mean Forecast', 'Naive Forecast', 'Drift Forecast'])

benchmarks_errors


# Export the train set e test set as csv to reuse them in the next chapter
close_train = pd.DataFrame(close_train)
close_train.to_csv('Project\\Thesis\\Train and test data\\close_train.csv')

close_test.to_csv('Project\\Thesis\\Train and test data\\close_test.csv')
close_test
