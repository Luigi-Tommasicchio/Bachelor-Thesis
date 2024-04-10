import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.tsa.stattools as sts #per l'.adfuller()
from statsmodels.tsa.seasonal import seasonal_decompose, STL, MSTL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
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
plot_acf(close_train, lags = 40, zero = False)
plt.title('ACF for Closing Price', size = 24)
plt.ylim(-1, 1.1)
plt.show()

# PACF PLOT
plot_pacf(close_train, lags = 40, zero = False, method = 'ols')
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

adf_test(close_diff)

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
axes[0, 0].plot(close_train, c='blue'); axes[0, 0].set_title('Serie Originale'); axes[0, 0].set_xlabel('Data'); axes[0, 0].set_ylabel('Prezzo $')
axes[0, 0].tick_params(axis='x', labelrotation = 45)

plot_acf(close_train, ax=axes[0, 1],auto_ylims=True, lags=40, zero=False, c='blue'); axes[0,1].set_xlabel('Lags')
acf_values, confint = acf(close_train, alpha=0.05, nlags=40)
lower_bound = confint[0:, 0] - acf_values[0:]
upper_bound = confint[0:, 1] - acf_values[0:]
lags = np.arange(0, len(acf_values[0:]))
ciao = []
for i in range(len(acf_values[0:])):
    if acf_values[i] > upper_bound[i]:
        ciao.append(acf_values[i])
    elif acf_values[i] < lower_bound[i]:
        ciao.append(acf_values[i])
    else:
        ciao.append('NaN')
axes[0,1].scatter(x=lags[1:], y=ciao[1:], zorder=3, c='orangered')

plot_pacf(close_train, ax=axes[0, 2],auto_ylims=True, lags=40, zero=False, method='ols', c='blue'); axes[0,2].set_xlabel('Lags')
pacf_values, confint = pacf(close_train, alpha=0.05, nlags=40, method='ols')
lower_bound = confint[1:, 0] - pacf_values[1:]
upper_bound = confint[1:, 1] - pacf_values[1:]
lags = np.arange(0, len(pacf_values[1:]))
ciao = []
for i in range(len(pacf_values[1:])):
    if pacf_values[i] > upper_bound[0]:
        ciao.append(pacf_values[i])
    elif pacf_values[i] < lower_bound[0]:
        ciao.append(pacf_values[i])
    else:
        ciao.append('NaN')
axes[0,2].scatter(x=lags[1:], y=ciao[1:], zorder=3, c='orangered')

# Differenziazione di 1° ordine
axes[1, 0].plot(close_diff, c='blue'); axes[1, 0].set_title('Differenziazione di 1° ordine'); axes[1, 0].set_xlabel('Data'); axes[1, 0].set_ylabel('Variazione giornaliera del prezzo (%)')
axes[1, 0].tick_params(axis='x', labelrotation = 45)

plot_acf(close_diff, ax=axes[1, 1],auto_ylims=True, lags=40, zero=False, c='blue'); axes[1,1].set_xlabel('Lags')
acf_values, confint = acf(close_diff, alpha=0.05, nlags=40)
lower_bound = confint[1:, 0] - acf_values[1:]
upper_bound = confint[1:, 1] - acf_values[1:]
lags = np.arange(0, len(acf_values[1:]))
ciao = []
for i in range(len(acf_values[1:])):
    if acf_values[i] > upper_bound[i]:
        ciao.append(acf_values[i])
    elif acf_values[i] < lower_bound[i]:
        ciao.append(acf_values[i])
    else:
        ciao.append('NaN')
axes[1,1].scatter(x=lags, y=ciao, zorder=3, c='orangered')

plot_pacf(close_diff, ax=axes[1, 2],auto_ylims=True, lags=40, zero=False, method='ols', c='blue'); axes[1,2].set_xlabel('Lags')
pacf_values, confint = pacf(close_diff, alpha=0.05, nlags=40, method='ols')
lower_bound = confint[1:, 0] - pacf_values[1:]
upper_bound = confint[1:, 1] - pacf_values[1:]
lags = np.arange(0, len(pacf_values[1:]))
ciao = []
for i in range(len(pacf_values[1:])):
    if pacf_values[i] > upper_bound[0]:
        ciao.append(pacf_values[i])
    elif pacf_values[i] < lower_bound[0]:
        ciao.append(pacf_values[i])
    else:
        ciao.append('NaN')
axes[1,2].scatter(x=lags, y=ciao, zorder=3, c='orangered')

fig.suptitle('Effetto della differenziazione', fontsize=16)
fig.tight_layout()
fig.subplots_adjust(hspace=0.4)
plt.show()

adf_test(close_train)
adf_test(close_diff)

############################################################################################################################
# Adesso sulla base del PACF della serie originale, provo a fittare un modello AUTOREGRESSIVO.
# Model Selection in base all'MSE e AIC
list_mse = []
aic = []
for p in range(1,15):
    model_ar = ARIMA(close_train, order=(p,0,0))
    model_ar_fit = model_ar.fit()
    forecasts = model_ar_fit.forecast(len(close_test['Close']))
    mean = close_test['Close'].mean()
    mse = sum((forecasts-mean)**2)/len(close_test['Close'])
    list_mse.append(mse)
    aic.append(model_ar_fit.aic)

fig, ax = plt.subplots(2, sharex=False)
list_mse = pd.Series(list_mse)
aic = pd.Series(aic)
min_mse = list_mse.idxmin()+1
min_aic = aic.idxmin()+1

ax[0].plot(range(1,15), list_mse, marker='o', color='blue'); ax[0].set_title('MSE'); ax[0].set_xlabel('Lags')
ax[0].scatter(min_mse, list_mse.min(), marker='o', color='red', lw=3, zorder=3)
ax[1].plot(range(1,15), aic, marker='o', color='blue'); ax[1].set_title('AIC'); ax[1].set_xlabel('Lags')
ax[1].scatter(min_aic, aic.min(), marker='o', color='red', lw=3, zorder=3)
fig.subplots_adjust(hspace=0.5)
plt.show()


# ARIMA FORECASTING
#prendo solo la colonna con i prezzi dal df test
close_test_1 = close_test['Close'].reset_index()

#questa è la serie a cui appendo i forecast
close_train_1 = close_train.copy()
close_train_1 = pd.Series(close_train_1.Close)

# Serie su cui andiamo a fittare il modello e serie a cui appenderemo le osservazioni vere
close_train_obs = close_train.copy().reset_index()['Close']
type(close_train_obs)
close_test_1 = close_test['Close']
type(close_test_1)

start = 0 # indice che mi serve per lo slice delle osservazioni da aggiungere per addestrare il modello
steps_ahead = 10 # di quanti step vogliamo procedere ogni volta

for i in range(int(len(close_test_1)/steps_ahead)):
    model_ar = ARIMA(close_train_obs, order=(10,0,0))
    model_ar_fit = model_ar.fit()
    forecasts = model_ar_fit.forecast(steps_ahead)
    foracasts = pd.Series(forecasts, index=forecasts.index, name='Close')
    close_train_1 = close_train_1._append(forecasts, ignore_index=True)# da plottare, questa contiene i forecast

    close_train_obs = close_train_obs._append(pd.Series(close_test_1.iloc[start:start+steps_ahead]), ignore_index=True) # la serie su cui si fitta il modello
    start += steps_ahead

print(len(close_train_1), len(close_train_obs))

fig, ax = plt.subplots(2,1, figsize=(15,7))
close_train_1.plot(label='Forecasted Prices', c='orangered', ax=ax[0]); ax[0].set_ylabel('Prezzi $')
close_train_obs.plot(label='Actual Prices', c='blue', ax=ax[0])
plt.title('Forecasts vs. Actual Prices')
plt.ylabel('Prezzi $')
plt.xlabel('Data')
fig.subplots_adjust(hspace=0.5)
plt.legend()
plt.show() 

########################################################################################################################################

# Iniziamo con la parte di ARIMA
# la serie di train è close_train, mentre quella di test è close_test['Close']
# prima di tutto parliamo di seasonal decomposition, poi magari famo un adf, poi siccome non è un cazzo stazionario
# magari famo una differenziarione, e devo vedè se riesco a farlo con il seasonal decompose a leva sto cazzo de tren per poi rimetterlo?
# [X] 1) parliamo dei modelli ARIMA, quindi introduco il concetto di autocorrelazione su cui si basano e parlo del plot acf,
# [X] 2) faccio il plot acf, perlo della non stazionarietà della serie storica, quindi faccio il seasonal decompose e rimuovo il trend.
# [X] 3) rimosso il trend faccio il test di stazionarietà e vedo che la serie è stazionaria e rifaccio il acf plot per mostrare la differenza.
# [ ] 4) parto nuovamenteo con l'acf e pacf e parlo dei modelli ar e ma, poi arma e via. 
# [ ] 6) il modello è fittato quindi sui residui
# [ ] 5) discutere dell'orizzonte temporale preso per il forecast e mostrare come con degli one step ahead forecast sono più precisi?



# 1) ACF della serie non stazionaria e adf test:
plot_acf(close_train,auto_ylims=True, lags=40, zero=False, c='blue')
acf_values, confint = acf(close_train, alpha=0.05, nlags=40)
lower_bound = confint[0:, 0] - acf_values[0:]
upper_bound = confint[0:, 1] - acf_values[0:]
lags = np.arange(0, len(acf_values[0:]))
ciao = []
for i in range(len(acf_values[0:])):
    if acf_values[i] > upper_bound[i]:
        ciao.append(acf_values[i])
    elif acf_values[i] < lower_bound[i]:
        ciao.append(acf_values[i])
    else:
        ciao.append('NaN')
plt.scatter(x=lags[1:], y=ciao[1:], zorder=3, c='orangered')
plt.ylabel('Autocorrelation Coefficient')
plt.xlabel('Lags')
plt.show() 

adf_test(close_train)

# 2) Decomposizione della serie per renderla stazionaria, rimuoviamo il trend e facciamo quindi adf e acf plot
decomposition = seasonal_decompose(close_train, model='additive') # Di default setta un periodo stazionale pari a 5 giorni ovvero una settimana di trading
decomposition.plot()                                              # restituisce un oggetto con vari attributi tipo trend, stagionalità e residui.
plt.show()                                                        # da qui praticamente capiamo che non esiste un effetto seasonal, ma che esiste solamente 
                                                                  # una componente di trend che possiamo rimuovere
trend = decomposition.trend                                                               
close_train_nt = close_train.Close - trend
close_train_nt.plot()
close_train_nt.isna().sum()
close_train_nt = close_train_nt.ffill().bfill()
plt.show()

adf_test(close_train_nt) 

plot_acf(close_train_nt,auto_ylims=True, lags=40, zero=True, c='blue')
acf_values, confint = acf(close_train_nt, alpha=0.05, nlags=40)
lower_bound = confint[0:, 0] - acf_values[0:]
upper_bound = confint[0:, 1] - acf_values[0:]
lags = []
ciao = []
for i in range(len(acf_values[0:])):
    if acf_values[i] > upper_bound[i]:
        ciao.append(acf_values[i])
        lags.append(i)
    elif acf_values[i] < lower_bound[i]:
        ciao.append(acf_values[i])
        lags.append(i)
    else:
        None
plt.scatter(lags, ciao, zorder=3, c='orangered')
plt.ylabel('Autocorrelation Coefficient')
plt.xlabel('Lags')
plt.show()                                                  # Magari si potrebbe fare il grafico figo con 4/6 sublplots con serie, acf e pacf per flexare.

# 4) Modello ar base base:
