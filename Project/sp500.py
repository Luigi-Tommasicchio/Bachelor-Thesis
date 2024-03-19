import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
from pmdarima.arima import auto_arima
from scipy.stats import chi2
import mplfinance as mpf



# Scarico i dati da Yahoo Finance
ticker = 'AAPL'
data = yf.Ticker(ticker).history(ticker, start = '2023-03-18', end = '2024-03-18', interval='1h')[['Open', 'Close', 'High', 'Low', 'Volume']]


# Formatto l'indice come Datetime
data = data.reset_index()
data['Datetime'] = pd.to_datetime(data['Datetime'])
data.set_index('Datetime', inplace=True)
print(type(data.index))

# Stampo le prime righe del Dataframe
print(data.head())

# Plotto i dati come Candlestick Chart
mpf.plot(data, type='candle',volume=True, style='charles', title='Candlestick Chart AAPL', ylabel='Prezzo', ylabel_lower='Volume',
         show_nontrading=False, mav = (10,50,200))



# Estrazione della serie temporale dei prezzi di chiusura
close= data['Close']
print(close)


# Esecuzione del test di Dickey-Fuller Aumentato (ADF) per verificare la stazionarietà
adf_result = adfuller(close)
print(adf_result)
# Visualizzazione dei risultati del test ADF
adf_output = pd.Series(adf_result[0:4], index=['Test Statistic','p-value','Lags Used','Number of Observations Used'])
for key,value in adf_result[4].items():
    adf_output[f'Critical Value ({key})'] = value
print('Augmented Dickey-Fuller Test:')
print(adf_output)


# Calculating daily returns
returns = np.log(close / close.shift(1)).dropna()

# Re-executing the Augmented Dickey-Fuller (ADF) test on daily returns
adf_result_returns = adfuller(returns)

# Displaying the ADF test results for daily returns
adf_output_returns = pd.Series(adf_result_returns[0:4], index=['Test Statistic','p-value','Lags Used','Number of Observations Used'])
for key, value in adf_result_returns[4].items():
    adf_output_returns[f'Critical Value ({key})'] = value

print('Augmented Dickey-Fuller Test:')
print(adf_output_returns)


# Generating the ACF and PACF plots for daily returns
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# ACF plot for daily returns
plot_acf(returns, ax=ax1, lags=40, title="ACF of Returns", auto_ylims=True)

# PACF plot for daily returns
plot_pacf(returns, ax=ax2, lags=40, method='ywm', title="PACF of Returns", auto_ylims=True)

plt.tight_layout()
plt.show()



# Divisione dei dati in training set (80%) e test set (20%)
train_size = int(len(returns) * 0.8)
train_returns, test_returns = returns[:train_size], returns[train_size:]

# Addestramento del modello ARIMA(1,0,1) sui rendimenti del training set
model_arima_101 = ARIMA(train_returns, order=(1,0,1)).fit()

# Visualizzazione del summary del modello
print(model_arima_101.summary())


# Addestramento del modello ARIMA(2,0,2) sui rendimenti del training set
model_arima_202 = ARIMA(train_returns, order=(2,0,2)).fit()

# Visualizzazione del summary del modello ARIMA(2,0,2)
print(model_arima_202.summary())

# Definizione della funzione per il test LLR
def LLR_test(mod1, mod2, DF=1):
    L1 = mod1.llf
    L2 = mod2.llf
    LR = 2 * (L2 - L1)
    p = chi2.sf(LR, DF)
    return p

# Esecuzione del test LLR tra i modelli ARIMA(1,0,1) e ARIMA(2,0,2)
p_value_llr = LLR_test(model_arima_101, model_arima_202, 2)
print(f"LLR Test P-Value: {p_value_llr}")

# Forecast returns for the test set from the ARIMA(2,0,2) model
forecast_returns_arima_202 = model_arima_202.forecast(steps=len(test_returns))

# Calculate the corrected predicted prices starting from the last known price in the training set
last_train_price = close_prices.iloc[train_size - 1]

# Calcolo dei prezzi previsti partendo dall'ultimo prezzo noto nel training set
# e applicando i rendimenti previsti uno per uno
predicted_prices = [last_train_price]

# Applichiamo ogni rendimento previsto cumulativamente
for return_forecast in forecast_returns_arima_202:
    new_predicted_price = predicted_prices[-1] * np.exp(return_forecast)
    predicted_prices.append(new_predicted_price)

# Rimuoviamo il primo elemento per allineare la lunghezza dei prezzi previsti con quella del test set
predicted_prices.pop(0)

# Assicuriamoci che la lunghezza dei prezzi previsti corrisponda a quella del test set
if len(predicted_prices) != len(test_returns):
    print(f"Adjusting lengths. Predicted: {len(predicted_prices)}, Test: {len(test_returns)}")
    predicted_prices_corrected = predicted_prices[:len(test_returns)]


# Creazione del grafico
plt.figure(figsize=(15, 8))

# Plot dei prezzi nel periodo di training
plt.plot(close_prices.index[:train_size], close_prices[:train_size], label='Training Prices', color='blue')

# Plot dei prezzi effettivi nel periodo di test
plt.plot(close_prices.index[train_size:], close_prices[train_size:], label='Test Prices', color='cyan')

# Plot dei prezzi previsti nel periodo di test
plt.plot(close_prices.index[train_size:train_size + len(predicted_prices)], predicted_prices, label='ARIMA(2,0,2) Predicted Prices', color='magenta', linestyle='--')

plt.title('Training, Actual, and Predicted Prices: ARIMA(2,0,2) Model')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()

# Calcolo dei residui dal modello ARIMA(2,0,2)
residuals_arima_202 = model_arima_202.resid

# Grafico dei residui nel tempo
plt.figure(figsize=(14, 7))
plt.plot(residuals_arima_202, color='blue', label='Residuals')
plt.title('Residuals of ARIMA(2,0,2) Model Over Time')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.legend()
plt.show()

# Istogramma dei residui
plt.figure(figsize=(7, 5))
plt.hist(residuals_arima_202, bins=30, color='blue', alpha=0.7, label='Residuals Distribution')
plt.title('Histogram of Residuals from ARIMA(2,0,2) Model')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# ACF plot dei residui
plt.figure(figsize=(10, 5))
plot_acf(residuals_arima_202, lags=40, alpha=0.05, title="ACF of Residuals from ARIMA(2,0,2) Model", auto_ylims=True)
plt.show()
