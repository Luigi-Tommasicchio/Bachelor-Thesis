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
close_test = pd.DataFrame(close_test)
# Benchmark mean
mean_forecast = close_train.mean().repeat(252)
close_test['mean_forecast'] = mean_forecast


# Benchmark Naive
naive_forecast = close_train[-1]  # Prendi l'ultimo valore del set di addestramento
close_test['naive_forecast'] = naive_forecast


# Benchmark Drift
drift = (close_train[-1] - close_train[0]) / len(close_train)  # Calcola il tasso di variazione medio

drift_forecast = []
for i in range(0, len(close_test)):
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




# Differenziare una serie e normalizzarla
import numpy as np
apple_diff = apple['Close'].diff()
apple_close = apple.Close[0] + apple_diff.cumsum()
apple_close.plot(label='Integrated')
apple['Close'].plot(label='original',linestyle='dotted', color='r', linewidth=2)
plt.legend()
plt.show()


