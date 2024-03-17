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

# Setto l'indice
df.set_index('Date', inplace = True)
df.head()

# Setto la frequenza come business days
df = df.asfreq('b') 
df.isna().sum()
df = df.ffill()
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








# MA(1) per i Returns
sgt.plot_acf(df.returns[1:], zero = False, lags = 40)
plt.title("ACF for Returns", size = 24)
plt.ylim(-0.2, 0.2)
plt.show()

model_ret_ma_1 = ARIMA(df.returns[1:], order = (0,0,1))
results_ret_ma_1 = model_ret_ma_1.fit()
results_ret_ma_1.summary()

model_ret_ma_2 = ARIMA(df.returns[1:], order = (0,0,2))
results_ret_ma_2 = model_ret_ma_2.fit()
print(results_ret_ma_2.summary())
print("\nLLR test p-value = " + str(LLR_test(model_ret_ma_1, model_ret_ma_2)))

model_ret_ma_3 = ARIMA(df.returns[1:], order = (0,0,3))
results_ret_ma_3 = model_ret_ma_3.fit()
print(results_ret_ma_3.summary())
print("\nLLR test p-value = " + str(LLR_test(model_ret_ma_2, model_ret_ma_3)))

model_ret_ma_4 = ARIMA(df.returns[1:], order = (0,0,4))
results_ret_ma_4 = model_ret_ma_4.fit()
print(results_ret_ma_4.summary())
print("\nLLR test p-value = " + str(LLR_test(model_ret_ma_3, model_ret_ma_4)))

model_ret_ma_5 = ARIMA(df.returns[1:], order = (0,0,5))
results_ret_ma_5 = model_ret_ma_5.fit()
print(results_ret_ma_5.summary())
print("\nLLR test p-value = " + str(LLR_test(model_ret_ma_4, model_ret_ma_5)))

model_ret_ma_6 = ARIMA(df.returns[1:], order = (0,0,6))
results_ret_ma_6 = model_ret_ma_6.fit()
print(results_ret_ma_6.summary())
print("\nLLR test p-value = " + str(LLR_test(model_ret_ma_5, model_ret_ma_6)))

model_ret_ma_7 = ARIMA(df.returns[1:], order = (0,0,7))
results_ret_ma_7 = model_ret_ma_7.fit()
print(results_ret_ma_7.summary())
print("\nLLR test p-value = " + str(LLR_test(model_ret_ma_6, model_ret_ma_7)))

model_ret_ma_8 = ARIMA(df.returns[1:], order = (0,0,8))
results_ret_ma_8 = model_ret_ma_8.fit()
print(results_ret_ma_8.summary())
print("\nLLR test p-value = " + str(LLR_test(model_ret_ma_7, model_ret_ma_8)))

model_ret_ma_9 = ARIMA(df.returns[1:], order = (0,0,9))
results_ret_ma_9 = model_ret_ma_9.fit()
print(results_ret_ma_9.summary())
print("\nLLR test p-value = " + str(LLR_test(model_ret_ma_8, model_ret_ma_9)))

LLR_test(model_ret_ma_1, model_ret_ma_9, DF=8)


# Vediamo i residui del modello ma(9)
df['res_ret_ma_9'] = results_ret_ma_9.resid[1:]
df.head()


print('The Mean of the residuals: ' + str(round(df.res_ret_ma_9.mean(),3)) + 
      '\nThe Variance of the Residuals: '+ str(round(df.res_ret_ma_9.var(),3)))
round(np.sqrt(df.res_ret_ma_9.var()),3)

df.res_ret_ma_9[1:].plot(figsize = (15,5))
plt.title("Residuals of Returns", size = 24)
plt.show()

adf_test(df.res_ret_ma_9[2:])

sgt.plot_acf(df.res_ret_ma_9[2:], zero = False, lags = 40)
plt.title("ACF Of Residuals for Returns",size=24)
plt.ylim(-0.2,0.2)
plt.show()

df.head()
