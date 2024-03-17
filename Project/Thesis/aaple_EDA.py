import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pylab
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.graphics.tsaplots as sgt
from scipy.stats.distributions import chi2

# Importo i dati
dati = pd.read_csv('C:\\Users\\luigi\\Desktop\\Thesis2\\Project\\Thesis\\AAPL.csv')

df = dati.copy()

print(df.head())
print(df.describe())
print(df.isna().sum())
print(df.info())

# Plotto i dati
df.plot(x='Date', y='Close', title="Apple Closing Price", figsize=(15,5))
plt.show()

# QQ Plot
scipy.stats.probplot(df.Close, plot = pylab)
pylab.show()

# Trasformo il dataframe in una time-series
df.Date = pd.to_datetime(df.Date).dt.date
print(df.head())
print(df.Date.describe())

# Setto la frequenza come business days
df = df.asfreq('b', method='ffill') 
df.head()

# Setto l'indice
df.set_index('Date', inplace = True)
df.head()
df.isna().sum()

# Split the data
size = int(len(df)*0.8)

df_train = pd.DataFrame(df.Close.iloc[:size])
df_test = pd.DataFrame(df.Close.iloc[size:])

# Controllo che lo split sia corretto
print(df_train.tail())
print(df_test.head())

# Check di stazionarietà
def adf_test(column):
    # Esecuzione del test di Dickey-Fuller Aumentato (ADF) per la colonna specificata
    adf_result = sts.adfuller(column)
    
    adf_output = pd.Series(adf_result[0:4], index=['Test Statistic','P-value','Lags Used','Number of Observations Used'])
    for key, value in adf_result[4].items():
        adf_output[f'Critical Value ({key})'] = value

    return adf_output

adf_test(df_train.Close)

# Seasonal decomposition
s_dec_additive = seasonal_decompose(df_train.Close, model = 'additive')
s_dec_additive.plot()
plt.show()

s_dec_multiplicative = seasonal_decompose(df_train.Close, model = 'multiplicative')
s_dec_multiplicative.plot()
plt.show()

# ACF PLOT
sgt.plot_acf(df_train.Close, lags = 40, zero = False)
plt.title('ACF for Closing Price', size = 24)
plt.ylim(-1, 1.1)
plt.show()

# PACF PLOT
sgt.plot_pacf(df_train.Close, lags = 40, zero = False, method = 'ols')
plt.title('PACF for Closing Price', size = 24)
plt.ylim(-0.2, 1.1)
plt.show()


# ciaoo


# AR(1) Model
from statsmodels.tsa.arima.model import ARIMA

model_ar = ARIMA(df.Close, order=(1,0,0))
results_ar = model_ar.fit()
results_ar.summary()

model_ar_2 = ARIMA(df.Close, order = (2,0,0))
results_ar_2 = model_ar_2.fit()
results_ar_2.summary()

model_ar_9 = ARIMA(df.Close, order = (9,0,0))
results_ar_9 = model_ar_9.fit()
results_ar_9.summary()

model_ar_10 = ARIMA(df.Close, order=[10,0,0])
results_ar_10 = model_ar_10.fit()
results_ar_10.summary()

model_ar_11 = ARIMA(df.Close, order=[11,0,0])
results_ar_11 = model_ar_11.fit()
results_ar_11.summary()

# LLR_test
def LLR_test(mod_1, mod_2, DF=1):
    L1 = mod_1.fit().llf
    L2 = mod_2.fit().llf
    LR = (2*(L2-L1))
    p = chi2.sf(LR, DF).round(3)
    return p

LLR_test(model_ar, model_ar_11, DF=10)


# Calcoliamo i returns
df['returns'] = df.Close.pct_change(1).mul(100)
df = df.iloc[1:]
adf_test(df.returns)

# ACF per i Returns
sgt.plot_acf(df.returns, lags = 40, zero = False)
plt.title("ACF Apple Returns", size = 24)
plt.ylim(-0.2, 0.2)
plt.show()

# PACF per i Returns
sgt.plot_pacf(df.returns, lags = 40, zero = False, method = 'ols')
plt.title("PACF Apple Returns", size = 24)
plt.ylim(-0.2, 0.2)
plt.show()

# AR(1) per i Returns
model_ret_ar_1 = ARIMA(df.returns, order = (1,0,0))
results_ret_ar_1 = model_ret_ar_1.fit()
results_ret_ar_1.summary()

model_ret_ar_2 = ARIMA(df.returns, order = (2,0,0))
results_ret_ar_2 = model_ret_ar_2.fit()
results_ret_ar_2.summary()

model_ret_ar_3 = ARIMA(df.returns, order = (3,0,0))
results_ret_ar_3 = model_ret_ar_3.fit()
results_ret_ar_3.summary()

model_ret_ar_7 = ARIMA(df.returns, order = (7,0,0))
results_ret_ar_7 = model_ret_ar_7.fit()
results_ret_ar_7.summary()

model_ret_ar_8 = ARIMA(df.returns, order = (8,0,0))
results_ret_ar_8 = model_ret_ar_8.fit()
results_ret_ar_8.summary()

model_ret_ar_9 = ARIMA(df.returns, order = (9,0,0))
results_ret_ar_9 = model_ret_ar_9.fit()
results_ret_ar_9.summary()

model_ret_ar_10 = ARIMA(df.returns, order = (10,0,0))
results_ret_ar_10 = model_ret_ar_10.fit()
results_ret_ar_10.summary()

LLR_test(model_ret_ar_1, model_ret_ar_9, DF=8)

# Esaminiamo i residui del modello AR miglior per i prezzi
df['res_price'] = results_ar_11.resid
df.res_price.mean()
df.res_price.var()
adf_test(df.res_price)

sgt.plot_acf(df.res_price, zero = False, lags = 40)
plt.title("ACF for price residuals", size = 24)
plt.ylim(-0.1, 0.1)
plt.show()

# Plottiamo i residui
df.res_price[1:].plot(figsize=(15,5))
plt.title("Residuals of Price", size=24)
plt.show()

# Esaminiamo i residui del modello AR miglior per i returns
df["res_ret"] = results_ret_ar_9.resid
df.res_ret.mean()
df.res_ret.var()
adf_test(df.res_ret)

sgt.plot_acf(df.res_ret, zero = False, lags = 40)
plt.title("ACF For Residuals of Returns", size = 24)
plt.ylim(-0.1, 0.1)
plt.show()

# Plottiamo i residui dei returns
df.res_ret[1:].plot(figsize=(15,5))
plt.title("Residuals of Returns", size=24)
plt.show()






# Formatto la data nel datatype corretto
df['Date'] = pd.to_datetime(df['Date'])

# Imposto la data come indice del dataframe
df = df.set_index('Date').asfreq('d', method='ffill')
print(df.head())

start_date = '2022-09-14'
end_date = '2023-12-29'

# Creo un nuovo dateframe che parte da start date e finisce a end date
new_df = df.loc[start_date:end_date]
print(new_df)

# Importo il csv con i valori della sentiment analysis
sentiment = pd.read_csv('aapl_sentiment.csv')
sentiment['Data'] = pd.to_datetime(sentiment['Data'])

sentiment = sentiment.set_index('Data').asfreq('d', method='ffill')

sentiment = sentiment.loc[start_date:end_date]

print(len(sentiment) == len(new_df))

new_df['Sentiment'] = sentiment['compound']
df = new_df
print(df)

df.to_csv('aapl_with_sentiment.csv')

