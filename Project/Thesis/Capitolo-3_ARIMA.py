import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.tsa.stattools as sts #per l'.adfuller()
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.graphics.tsaplots as sgt
from scipy.stats.distributions import chi2
from statsmodels.tsa.arima.model import ARIMA 

close_train = pd.read_csv('Project\\Thesis\\Train and test data\\close_train.csv', parse_dates=True, index_col='Date')
close_test = pd.read_csv('Project\\Thesis\\Train and test data\\close_test.csv', parse_dates=True, index_col='Date')[['Close', 'drift_forecast']]

close_train.head()
close_test.head()

#Setto la frequenza dei giorni come business days
close_train = close_train.asfreq('b')
close_train.isna().sum()
close_train = close_train.fillna(method = 'ffill')
close_train.isna().sum()

close_test = close_test.asfreq('b')
close_test.isna().sum()
close_test = close_test.fillna(method = 'ffill')
close_test.isna().sum()

close_train.plot(figsize=(15,7), label='Close Train', color='blue')
close_test.Close.plot(label='Close Test', color='red')
close_test.drift_forecast.plot(label='Drift Forecast', color='purple')
plt.ylabel('Prezzo $')
plt.xlabel('Data')
plt.title('Prezzi di chiusura AAPL')
plt.legend()
plt.show()

# Test di stazionarietà
def adf_test(column):
    # Esecuzione del test di Dickey-Fuller Aumentato (ADF) per la colonna specificata
    adf_result = sts.adfuller(column)
    
    adf_output = pd.Series(adf_result[0:4], index=['Test Statistic','P-value','Lags Used','Number of Observations Used'])
    for key, value in adf_result[4].items():
        adf_output[f'Critical Value ({key})'] = value

    return adf_output

adf_test(close_train)

# Seasonal decomposition
s_dec_additive = seasonal_decompose(close_train, model = 'additive')
s_dec_additive.plot()
plt.show()

s_dec_multiplicative = seasonal_decompose(close_train, model = 'multiplicative')
s_dec_multiplicative.plot()
plt.show()

# ACF PLOT
sgt.plot_acf(close_train, lags = 40, zero = False)
plt.title('ACF for Closing Price', size = 24)
plt.ylim(-1, 1.1)
plt.show()

# PACF PLOT
sgt.plot_pacf(close_train, lags = 40, zero = False, method = 'ols')
plt.title('PACF for Closing Price', size = 24)
plt.ylim(-0.2, 1.1)
plt.show()
                                
############################################################################################################################
# Differenzio la serie storica per poi integrarla.
close_diff = ((close_train.diff()/close_train.shift(1))*100).fillna(value=0)
close_diff.plot()
plt.ylabel('Variazione %')
plt.xlabel('Data')
plt.title('Rendimenti giornalieri percentuali (%)')
plt.legend()
plt.show()

# Dalla differenziata torno alla serie dei prezzi.
close_int = close_train.Close[0] + ((close_diff/100)*close_train.shift(1)).cumsum()
close_int.Close[0] = close_train.Close[0]
close_int; close_train

# Le plotto per vedere se combaciano.
plt.plot(close_int.index, close_int.Close, color='red', label='Serie modificata')
plt.plot(close_int.index, close_train.Close, color='blue', linestyle=':', label='Serie originale')
plt.title(f'Prezzi di chiusura giornalieri AAPL dal: {close_train.index.date.min()} al {close_train.index.date.max()}')
plt.ylabel('Prezzi $')
plt.xlabel('Data')
plt.legend()
plt.show()
############################################################################################################################
# Creo una figura in cui plotto la serie di train originale, quella differenziata e per ciascuna l'ACF ed il PACF
plt.rcParams.update({'figure.figsize':(15,10)})
fig, axes = plt.subplots(2, 3, sharex=False)

# Serie Originale
axes[0, 0].plot(close_train); axes[0, 0].set_title('Serie Originale'); axes[0, 0].set_xlabel('Data'); axes[0, 0].set_ylabel('Prezzo $')
axes[0, 0].tick_params(axis='x', labelrotation = 45)
sgt.plot_acf(close_train, ax=axes[0, 1],auto_ylims=True, lags=40, zero=False)
sgt.plot_pacf(close_train, ax=axes[0, 2],auto_ylims=True, lags=40, zero=False)
axes[0,1].set_xlabel('Lags'); axes[0,2].set_xlabel('Lags')

# Differenziazione di 1° ordine
axes[1, 0].plot(close_diff); axes[1, 0].set_title('Differenziazione di 1° ordine'); axes[1, 0].set_xlabel('Data'); axes[1, 0].set_ylabel('Variazione giornaliera del prezzo (%)')
axes[1, 0].tick_params(axis='x', labelrotation = 45)
sgt.plot_acf(close_diff, ax=axes[1, 1],auto_ylims=True, lags=40, zero=False)
sgt.plot_pacf(close_diff, ax=axes[1, 2],auto_ylims=True, lags=40, zero=False)
axes[1,1].set_xlabel('Lags'); axes[1,2].set_xlabel('Lags')

fig.suptitle('Effetto della differenziazione', fontsize=16)
fig.tight_layout()
fig.subplots_adjust(hspace=0.4)
plt.show()

adf_test(close_train)
adf_test(close_diff)
############################################################################################################################
# Adesso sulla base del PACF della serie originale, provo a fittare un modello AUTOREGRESSIVO.
model_ar = ARIMA(close_train, order=(1,0,0))
model_ar_fit = model_ar.fit()
print(model_ar_fit.summary())

residuals = pd.DataFrame(model_ar_fit.resid)

plt.rcParams.update({'figure.figsize':(15,7)})
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0]); ax[0].set_ylim(-15,15)
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

forecasts = model_ar_fit.forecast(1)
close_test[:1].plot()
prediction.plot()
plt.show()

type(prediction)





close_test_1 = close_test['Close'].reset_index()
close_train_1 = close_train.copy().reset_index()

model_ar = ARIMA(close_train_1.Close, order=(1,0,0))
model_ar_fit = model_ar.fit()
forecasts = model_ar_fit.forecast(1)
forecasts = forecasts.reset_index()
close_train_1 = close_train_1._append(forecasts, ignore_index=True)


close_train_1 = close_train_1.set_index('Date')
close_train_1.plot()
plt.show()

len(close_train_1)
len(close_test_1)
close_test_
plt.show()
def arima_forecast(series):
    sgt.acf(series)
