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
s_dec_additive = seasonal_decompose(close_train, model = 'additive', period=252)
s_dec_additive.plot()
plt.show()

s_dec_multiplicative = seasonal_decompose(close_train, model = 'multiplicative', period=252)
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