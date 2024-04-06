# Importo le librerie necessarie
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Download dei dati da Yahoo Finance
ticker = 'AAPL'
start_date = '2019-01-01'
end_date = '2024-01-01'

apple = yf.Ticker(ticker).history(ticker, start=start_date, end=end_date, interval='1d')[['Open', 'Close', 'High', 'Low', 'Volume']]

apple.head()

# Controllo il datatype delle colonne
apple.info()
# Controllo se ci sono valori nulli
apple.isna().sum()

# Riformatto l'indice da 'datetime' a 'date'
apple = apple.reset_index()
apple['Date'] = pd.to_datetime(apple['Date'].dt.date)
apple.set_index('Date', inplace=True)
apple.head()

apple.info()

# Plotto i dati come Candlestick Chart
import mplfinance as mpf

fig, ax = mpf.plot(apple, type='candle',volume=True, style='yahoo', 
                ylabel='Prezzo', ylabel_lower='Volume', xlabel = 'Data', show_nontrading=False, 
                figratio=(4,3), figscale=1.2, xrotation=45, returnfig=True)
ax[0].set_title('Candlestick chart AAPL', fontsize=16, loc='center')
plt.show()


# Calcolo delle correlazioni tra le colonne del dataframe
corr = apple.select_dtypes('number').corr()

# Creazione del grafico a matrice di correlazione
import seaborn as sns

# Creazione del heatmap delle correlazioni con Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='Blues', fmt=".2f")
sns.set_theme(font_scale=1.5)
plt.grid(None)
plt.title('Heatmap delle Correlazioni', size=30, y=1.02)
plt.show()

# Estraggo la serie del prezzo di chiusura
close = apple['Close']
close.head()
close.info()

close.plot(figsize = (15,7), color='blue', label='Prezzo di Chiusura')
plt.title('Prezzi di Chiusura AAPL (2019-2024)', size=24, pad=15)
plt.yticks(size=15)
plt.xticks(size=15)
plt.xlim(close.index.min(), close.index.max())
plt.ylabel('Prezzo $', size=20, labelpad=13)
plt.xlabel('Data', size=20, labelpad=13)
plt.show()

# Divido i dati in train e test
split = int(len(close)*0.8)

close_train = close[:split]
close_test = close[split:]

close_train.tail(2)
close_test.head(2)

print('Dimensione set di Train: ' + str(len(close_train)) + 
      '\nDimensione set di Test: ' + str(len(close_test)))


close_train.plot(label='Train set', color='blue', figsize=(15,7))
close_test.plot(label='Test set', color='red')
plt.yticks(size=15)
plt.xticks(size=15)
plt.xlim(close.index.min(), close.index.max())
plt.ylabel('Prezzo $', size=20, labelpad=13)
plt.xlabel('Data', size=20, labelpad=13)
plt.title('Prezzi di chiusura AAPL', size=24, pad=15)
plt.legend()
plt.show()


# Creo tre benchmark per l'analisi:
close_test = close_test.reset_index()
close_test = pd.DataFrame({'Close':close_test.Close,
                           'Date':close_test.Date})
close_test = close_test.set_index('Date')

# Benchmark mean
mean_forecast = close_train.mean().repeat(252)
close_test['mean_forecast'] = mean_forecast

# Benchmark Naive
naive_forecast = close_train[-1]  # Prendi l'ultimo valore del set di addestramento
close_test['naive_forecast'] = naive_forecast

# Benchmark Drift
drift = (close_train[-1] - close_train[0]) / len(close_train)  # Calcola il tasso di variazione medio
drift
drift_forecast = []
for i in range(len(close_test)):
    forecast = close_train[-1] + (drift * i)
    drift_forecast.append(forecast)

close_test['drift_forecast'] = drift_forecast

# Plotto i Benchmark
import matplotlib.patheffects as pe

close_train.plot(label='Train set', color='blue', figsize=(15,7))
close_test.Close.plot(label='Test set', color='red')
close_test.naive_forecast.plot(label='Naive Forecast', color='green', lw=2)
close_test.drift_forecast.plot(label='Drift Forecast', color='purple', lw=2)
close_test.mean_forecast.plot(label='Mean Forecast', color='gold', lw=2, path_effects=[pe.Stroke(linewidth=2.5, foreground='black'), pe.Normal()])
plt.ylabel('Prezzo $')
plt.xlabel('Data')
plt.title('Prezzi di chiusura AAPL')
plt.legend()
plt.show()

# Calcolo il varie misure di errote per le varie previsioni:
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


# Esporto il train set e test set come csv per riutilizzarli in altri file
close_train = pd.DataFrame(close_train)
close_train.to_csv('Project\\Thesis\\Train and test data\\close_train.csv')

close_test.to_csv('Project\\Thesis\\Train and test data\\close_test.csv')
close_test
