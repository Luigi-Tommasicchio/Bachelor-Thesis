import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.tsa.stattools as sts #per l'.adfuller()
from statsmodels.tsa.seasonal import seasonal_decompose, STL, MSTL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from scipy.stats.distributions import chi2
from statsmodels.tsa.arima.model import ARIMA 

def LLR_test(mod_1, mod_2, DF = 1):
    L1 = mod_1.fit().llf
    L2 = mod_2.fit().llf
    LR = (2*(L2-L1))    
    p = chi2.sf(LR, DF).round(3)
    return p

close_train = pd.read_csv('Project\\Thesis\\Train and test data\\close_train.csv', parse_dates=True, index_col='Date')
close_test = pd.read_csv('Project\\Thesis\\Train and test data\\close_test.csv', parse_dates=True, index_col='Date')[['Close', 'drift_forecast']]

close_train.head()
close_test.head()


#Setto la frequenza dei giorni come business days
close_train.index = pd.date_range(start=close_train.index[0], periods=len(close_train), freq='B')
close_train.isna().sum()
close_test.index = pd.date_range(start=close_test.index[0], periods=len(close_test), freq='B')
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

############################################################################################################################

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
# [X] 1) parliamo dei modelli ARIMA, quindi introduco il concetto di autocorrelazione su cui si basano e parlo del plot acf, quindi faccio anche il adf
# [X] 2) faccio anche il seasonal decompose per parlare del fatto che non c'è seasonalità 
# [X] 3) differenziata la serie faccio il test di stazionarietà e vedo che la serie è stazionaria e rifaccio il acf plot per mostrare la differenza.
# [X] 4) parto nuovamente con l'acf e pacf. 
# [ ] 5) Modello AR
# [ ] 6) Modello MA
# [ ] 7) Modello ARMA
# [ ] 8) discutere dell'orizzonte temporale preso per il forecast e mostrare come con degli one step ahead forecast sono più precisi?



# Capitolo 3: Modellazione
# 1) ACF della serie non stazionaria e adf test:
adf_test(close_train)

fig, ax = plt.subplots(1,2, figsize=(20,7))
ax[0].plot(close_train, color='blue'); ax[0].set_ylabel('Prezzo $', size = 20); ax[0].set_xlabel('Data', size=20); ax[0].set_title('Serie \'close_train\'', size=20)
plot_acf(close_train,auto_ylims=True, lags=40, zero=True, c='blue', ax=ax[1]); ax[1].set_title('ACF Plot - \'close_train\'', size=20)
plt.ylabel('Coefficiente di autocorrelazione', size=20)
plt.xlabel('Lags', size=20)
plt.show() 

# 2) seasonal decompose per mostrare che comunque non c'è stagionalità:
seasonal_decomposition = seasonal_decompose(close_train, extrapolate_trend=1)
seasonal_decomposition.plot()
plt.show()

# 3) Differenzio la serie e faccio nuovamente il test di stazionarietà:
close_train_diff = ((close_train.diff()/close_train.shift(1))*100)[1:]
close_test_diff = ((close_test['Close'].diff()/close_test['Close'].shift(1))*100).bfill()

plt.rcdefaults()
close_train_diff.plot(c='blue', figsize=(15,7), label='close_train')
close_test_diff.plot(c='red', label='close_test')
plt.ylabel('Variazione %', size=20)
plt.xlabel('Data', size=20)
plt.title('Rendimenti giornalieri percentuali (%)', size=20)
plt.legend()
plt.show()

adf_test(close_train_diff)

# Dalla differenziata torno alla serie dei prezzi.
close_int = close_train.Close[0] + ((close_train_diff/100)*close_train.shift(1)).cumsum()
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

# Capitolo 3.1: Modello Autoregressivo AR(p):
# 4) ACF E PACF DELLA SERIE DIFFERENZIATA vs la serie non differenziata.
# Creo una figura in cui plotto la serie di train originale, quella differenziata e per ciascuna l'ACF ed il PACF
plt.rcdefaults()
plt.rcParams.update({'figure.figsize':(14,6)})
fig, axes = plt.subplots(1, 2, sharex=False)

# Differenziazione di 1° ordine
# plotto a sinistra l'acf:
plot_acf(close_train_diff, ax=axes[0],auto_ylims=False, lags=40, zero=True, c='blue', title=None); axes[0].set_xlabel('Lag', size=15)
acf_values, confint = acf(close_train_diff, alpha=0.05, nlags=40)
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
axes[0].scatter(x=lags, y=ciao, zorder=3, c='orangered')

# plotto a destra il pacf:
plot_pacf(close_train_diff, ax=axes[1],auto_ylims=False, lags=40, zero=True, method='ols', c='blue', title=None); axes[1].set_xlabel('Lag', size=15)
pacf_values, confint = pacf(close_train_diff, alpha=0.05, nlags=40, method='ols')
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
axes[1].scatter(x=lags, y=ciao, zorder=3, c='orangered')

# qui sistemo un po' di parametri per sistemare i grafici:
axes[0].set_ylim(-0.2, 1.05)
axes[1].set_ylim(-0.2, 1.05)
axes[0].set_title("Autocorrelation Function", fontsize=18)
axes[1].set_title("Partial Autocorrelation Function", fontsize=18)
fig.suptitle('ACF e PACF per la serie di addestramento differenziata', fontsize=20, y=1)
plt.show()

# 5) Modello AR(9): 
# Adesso sulla base del PACF della serie originale, provo a fittare un modello AUTOREGRESSIVO.
# Model Selection in base all'MSE e AIC
list_mse = []
aic = []
for p in range(1,15):
    model_ar = ARIMA(close_train_diff, order=(p,0,0))
    model_ar_fit = model_ar.fit()
    forecasts = model_ar_fit.forecast(len(close_test_diff))
    mean = close_test_diff.mean()
    mse = sum((forecasts-mean)**2)/len(close_test_diff)
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


# Scegliamo il modello AR avendo esso il minor AIC
# modello AR 1
model_ar_1 = ARIMA(close_train_diff[1:],order=(1,0,0))
model_ar_1_fit = model_ar_1.fit()
model_ar_1_fit.summary()

residui_ar_1 = model_ar_1_fit.resid
var = residui_ar_1.var()
sd = np.sqrt(var)
mean = residui_ar_1.mean()

# Analisi grafica dei residui: 
plt.rcParams.update({'figure.figsize':(15,7)})
fig, ax = plt.subplots(1,2)
residui_ar_1.plot(ax=ax[0], c='blue'); ax[0].set_title('Time plot', size=18); ax[0].set_xlabel('Data', size=16); ax[0].set_ylabel('Rendimento %', size=16)
residui_ar_1.plot(kind='hist', bins=50, ax=ax[1], color='blue'); ax[1].set_title('Istogramma', size=18); ax[1].set_xlabel('Rendimento %', size=16); ax[1].set_ylabel('Frequenza', size=16)
ax[0].axhline(y=mean, color='red', linestyle='--', label=r'$\mu$', lw=2)
ax[1].axvline(x=sd, color='red', linestyle='--', label=r'+1 $\sigma$', lw=2)
ax[1].axvline(x=-sd, color='red', linestyle='--', label=r'-1 $\sigma$', lw=2)
ax[1].axvline(x=mean, color='orange', linestyle='--', label=r'$\mu$', lw=2)
ax[0].legend()
ax[1].legend()
fig.suptitle('Analisi grafica dei residui del modello AR(1):', size=20)
plt.show()

plt.rcParams.update({'figure.figsize':(10,7)})
plot_acf(residui_ar_1,auto_ylims=True, lags=40, zero=False, c='blue')
acf_values, confint = acf(residui_ar_1, alpha=0.05, nlags=40)
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
plt.scatter(x=lags, y=ciao, zorder=3, c='orangered', lw=2)
plt.xlabel('Lag', size=18)
plt.xticks(size=14)
plt.yticks(size=14)
plt.title('ACF per i residui del modello AR(1)', size=20)
plt.show()

# Ciclo FOR per addestrare modelli AR con ordini da 1 a 15 e calcolare le metriche
aic = []
sse = []

for i in range(1, 21):
    model = ARIMA(close_train_diff[1:], order=(i,0,0)).fit()
    aic.append(model.aic)
    sse.append(model.sse)
# Visualizzazione delle metriche
fig, ax = plt.subplots(1,2, figsize=(13, 5))
plt.subplots_adjust(hspace=.5, wspace=.3)
ax[0].plot(range(1, 21), aic, label='AIC', color='blue', lw=2); ax[0].set_xlabel('Lags', size=14); ax[0].set_title('Akaike Information Criterion', size=16)
ax[1].plot(range(1, 21), sse, label='SSE', color='blue', lw=2); ax[1].set_xlabel('Lags', size=14); ax[1].set_title('Sum of Squared Errors', size=16)
ax[0].scatter(9, aic[8], zorder=3, color='red', lw=4)
fig.suptitle('Metriche di valutazione dei modelli AR(p)', size=20, y=1.0)
ax[0].set_xticks(range(1, 21))
ax[1].set_xticks(range(1, 21))
plt.show()

# Trovato il minimo dell'aic per il lag 9, si procede quindi al fitting del modello nuovo con la stampa del summary.
model_ar_9 = ARIMA(close_train_diff[1:], order=(9,0,0))
model_ar_9_fit = model_ar_9.fit()
model_ar_9_fit.summary()

residui_ar_9 = model_ar_9_fit.resid
var = residui_ar_9.var()
sd = np.sqrt(var)
mean = residui_ar_9.mean()

plt.rcParams.update({'figure.figsize':(15,7)})
fig, ax = plt.subplots(1,2)
residui_ar_9.plot(ax=ax[0], c='blue'); ax[0].set_title('Time plot', size=18); ax[0].set_xlabel('Data', size=16); ax[0].set_ylabel('Rendimento %', size=16)
residui_ar_9.plot(kind='hist', bins=50, ax=ax[1], color='blue'); ax[1].set_title('Istogramma', size=18); ax[1].set_xlabel('Rendimento %', size=16); ax[1].set_ylabel('Frequenza', size=16)
ax[0].axhline(y=mean, color='red', linestyle='--', label=r'$\mu$', lw=2)
ax[1].axvline(x=sd, color='red', linestyle='--', label=r'+1 $\sigma$', lw=2)
ax[1].axvline(x=-sd, color='red', linestyle='--', label=r'-1 $\sigma$', lw=2)
ax[1].axvline(x=mean, color='orange', linestyle='--', label=r'$\mu$', lw=2)
ax[0].legend()
ax[1].legend()
fig.suptitle('Analisi grafica dei residui del modello AR(9):', size=20)
plt.show()

fig, ax = plt.subplots(1,2)
plt.rcParams.update({'figure.figsize':(15,7)})
plot_acf(residui_ar_1,auto_ylims=True, lags=40, zero=False, c='blue', ax=ax[0])
acf_values, confint = acf(residui_ar_1, alpha=0.05, nlags=40)
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
ax[0].scatter(x=lags, y=ciao, zorder=3, c='orangered', lw=2)
plot_acf(residui_ar_9,auto_ylims=True, lags=40, zero=False, c='blue', ax=ax[1])
acf_values, confint = acf(residui_ar_9, alpha=0.05, nlags=40)
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
ax[1].scatter(x=lags, y=ciao, zorder=3, c='orangered', lw=2)
fig.suptitle('ACF per i residui', size=20)
ax[0].set_title('ACF Residui AR(1)', size=16)
ax[1].set_title('ACF Residui AR(9)', size=16)
ax[0].set_xlabel('Lags', size=14)
ax[1].set_xlabel('Lags', size=14)
plt.show()





# Modello MA(q)
plt.rcdefaults()
plt.rcParams.update({'figure.figsize':(14,6)})
fig, ax = plt.subplots(1, 2, sharex=False)

# Differenziazione di 1° ordine
# plotto a sinistra l'acf:
plot_acf(close_train_diff, ax=ax[0],auto_ylims=False, lags=40, zero=True, c='blue', title=None); ax[0].set_xlabel('Lags', size=15)
acf_values, confint = acf(close_train_diff, alpha=0.05, nlags=40)
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
ax[0].scatter(x=lags, y=ciao, zorder=3, c='orangered')

# plotto a destra il minimo dell'aic:
aic = []
for i in range(1, 21):
    model = ARIMA(close_train_diff[1:], order=(0,0,i)).fit()
    aic.append(model.aic)
# Visualizzazione delle metriche
ax[1].plot(range(1, 21), aic, label='AIC', color='blue', lw=2); ax[1].set_xlabel('Lags', size=14); ax[1].set_title('Akaike Information Criterion', size=16)
ax[1].scatter(14, aic[13], zorder=3, color='red', lw=4)
ax[1].set_xticks(range(1, 21))
ax[0].set_ylim(-0.2, 1.05)
ax[0].set_title("ACF plot", fontsize=18)
fig.suptitle('ACF e AIC per la scelta dell\'ordine q', fontsize=20, y=1)
plt.show()


# Fittiamo il modello ma(14)
model_ma_14 = ARIMA(close_train_diff[1:], order=(0,0,14))
model_ma_14_fit = model_ma_14.fit()
model_ma_14_fit.summary()


residui_ma_14 = model_ma_14_fit.resid
var = residui_ma_14.var()
sd = np.sqrt(var)
mean = residui_ma_14.mean()

plt.rcParams.update({'figure.figsize':(15,7)})
fig, ax = plt.subplots(1,2)
residui_ma_14.plot(ax=ax[0], c='blue'); ax[0].set_title('Time plot', size=18); ax[0].set_xlabel('Data', size=16); ax[0].set_ylabel('Rendimento %', size=16)
residui_ma_14.plot(kind='hist', bins=50, ax=ax[1], color='blue'); ax[1].set_title('Istogramma', size=18); ax[1].set_xlabel('Rendimento %', size=16); ax[1].set_ylabel('Frequenza', size=16)
ax[0].axhline(y=mean, color='red', linestyle='--', label=r'$\mu$', lw=2)
ax[1].axvline(x=sd, color='red', linestyle='--', label=r'+1 $\sigma$', lw=2)
ax[1].axvline(x=-sd, color='red', linestyle='--', label=r'-1 $\sigma$', lw=2)
ax[1].axvline(x=mean, color='orange', linestyle='--', label=r'$\mu$', lw=2)
ax[0].legend()
ax[1].legend()
fig.suptitle('Analisi grafica dei residui del modello AR(9):', size=20)
plt.show()

plt.rcParams.update({'figure.figsize':(10,7)})
plot_acf(residui_ma_14,auto_ylims=True, lags=40, zero=False, c='blue')
plt.xlabel('Lags', size=16)
plt.title('ACF dei residui del modello MA(14)', size=20)
plt.show()
























# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!           COME PLOTTARE LE PREVISIONI CON TANTO DI INTERVALLO DI CONFIDENZA            !!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Plot delle previsioni e degli intervalli di confidenza
fig, ax = plt.subplots(figsize=(10, 6))

# Plotta i dati osservati (se disponibili)
ax.plot(close_train_diff, label='Dati osservati', color='blue')

# Plotta le previsioni
forecast = model_ar_1_fit.forecast(steps=100)  # Numero di passi previsti nel futuro
ax.plot(forecast, label='Previsioni', color='red')

# Plotta gli intervalli di confidenza
forecast_ci = model_ar_1_fit.get_forecast(steps=100).conf_int()  # Calcola gli intervalli di confidenza
ax.fill_between(x=forecast_ci.index, y1=forecast_ci.iloc[:,0], y2=forecast_ci.iloc[:,1], color='lightgrey')

# Aggiungi etichette, legenda, titolo, ecc.
ax.set_xlabel('Tempo')
ax.set_ylabel('Valore')
ax.legend()
plt.title('Previsioni AR(1) con intervalli di confidenza')

# Mostra il grafico
plt.show()

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


















model_ar_7 = ARIMA(close_train_diff[1:], order=(7,0,0))
model_ar_7_fit = model_ar_7.fit()
model_ar_7_fit.summary()

LLR_test(model_ar_1, model_ar_7, DF=6)

model_ar_8 = ARIMA(close_train_diff[1:], order=(8,0,0))
model_ar_8_fit = model_ar_8.fit()
model_ar_8_fit.summary()

LLR_test(model_ar_1, model_ar_8, DF=7)

model_ar_9 = ARIMA(close_train_diff[1:], order=(9,0,0))
model_ar_9_fit = model_ar_9.fit()
model_ar_9_fit.summary()

LLR_test(model_ar_8, model_ar_9, DF=1)
LLR_test(model_ar_1, model_ar_9, DF=8)

model_ar_10 = ARIMA(close_train_diff[1:], order=(10,0,0))
model_ar_10_fit = model_ar_10.fit()
model_ar_10_fit.summary()

model_ar_10_fit.resid.plot()
plt.show()


LLR_test(model_ar_9, model_ar_10, DF=1)
LLR_test(model_ar_1, model_ar_10, DF=9)




forecasts_ar = model_ar_1_fit.forecast(len(close_test_diff))

model_ar_9 = ARIMA(close_train_diff[1:], order=(9,0,0)) # fittiamo sul test di training
model_ar_9_fit = model_ar_9.fit()
model_ar_9_fit.summary()
forecasts_ar = model_ar_9_fit.forecast(len(close_test_diff)) # forecast per il numero di osservazioni presenti nel test set

date_range = pd.date_range(start=close_test.index.min(), periods=len(forecasts_ar), freq='B')
forecasts_ar = pd.Series(forecasts_ar, index=date_range)
plt.plot(forecasts_ar)
plt.plot(close_test_diff)
plt.show()

# ADESSO trasformo i forecast appena effettuati in prezzi effettivi in dollari e li compariamo con il test set
forecasted_price_ar = []
start_price = close_train.Close[-1]
forecasted_price_ar.append(start_price)

for i in range(len(forecasts)):
    new_price = forecasted_price_ar[-1] + ((forecasts[i])/100) * forecasted_price_ar[-1]
    forecasted_price_ar.append(new_price)

date_range = pd.date_range(start=close_test.index.min(), periods=len(forecasted_price_ar)-1, freq='B')

# Creare una serie Pandas con i prezzi e le date come indice
forecasted_price_ar = pd.Series(forecasted_price_ar[1:], index=date_range)

plt.plot(forecasted_price_ar)
close_train.Close.plot()
close_test.Close.plot()
close_test.drift_forecast.plot()
plt.show()   


################################################################################################################################################
# ARIMA FORECASTING with custom steps-ahead
close_train_diff_copy = close_train_diff.copy()

start = 0 # indice che mi serve per lo slice delle osservazioni da aggiungere per addestrare il modello
steps_ahead = 10 # di quanti step vogliamo procedere ogni volta
forecasts_list = []

for i in range(int(len(close_test_diff)/steps_ahead)):
    model_ar_9_step = ARIMA(close_train_diff_copy.Close, order=(9,0,0))
    model_ar_9_step_fit = model_ar_9_step.fit()
    forecasts = model_ar_9_step_fit.forecast(steps_ahead)
    for i in forecasts:
        forecasts_list.append(i)

    close_train_diff_copy = close_train_diff_copy.Close._append(pd.Series(close_test_diff[start:start+steps_ahead]), ignore_index=True)
    close_train_diff_copy = pd.DataFrame(close_train_diff_copy)# la serie su cui si fitta il modello

    start += steps_ahead

forecasted_price = []
start_price = close_train.Close[-1]
forecasted_price.append(start_price)

for i in range(len(forecasts_list)):
    new_price = forecasted_price[-1] + ((forecasts_list[i])/100) * forecasted_price[-1]
    forecasted_price.append(new_price)

date_range = pd.date_range(start=close_test.index.min(), periods=len(forecasted_price)-1, freq='B')
forecasted_price = pd.Series(forecasted_price[1:], index=date_range)

plt.plot(forecasted_price)
close_train.Close.plot()
close_test.Close.plot()
close_test.drift_forecast.plot()
plt.show()   

# 5.1) Analisi dei residui del modello AR(9): 

residui = model_ar_9_fit.resid
residui.mean()
residui.var()
adf_test(residui)

plot_acf(residui, zero=False, auto_ylims=True, lags=40)
plt.show()

residui.plot()
plt.show()

#################################################################################################################################################

plot_acf(close_train_diff, zero=False, auto_ylims=True, lags=40)
plt.show()

model_ma_1 = ARIMA(close_train_diff[1:], order = (0,0,1))
model_ma_1_fit = model_ma_1.fit()
model_ma_1_fit.summary()

model_ma_2 = ARIMA(close_train_diff[1:], order = (0,0,2))
model_ma_2_fit = model_ma_2.fit()
print(model_ma_2_fit.summary())
print("\nLLR test p-value = " + str(LLR_test(model_ma_1, model_ma_2)))

model_ma_3 = ARIMA(close_train_diff[1:], order = (0,0,3))
model_ma_3_fit = model_ma_3.fit()
print(model_ma_3_fit.summary())
print("\nLLR test p-value = " + str(LLR_test(model_ma_2, model_ma_3)))

model_ma_4 = ARIMA(close_train_diff[1:], order = (0,0,4))
model_ma_4_fit = model_ma_4.fit()
print(model_ma_4_fit.summary())
print("\nLLR test p-value = " + str(LLR_test(model_ma_3, model_ma_4)))

model_ma_5 = ARIMA(close_train_diff[1:], order = (0,0,5))
model_ma_5_fit = model_ma_5.fit()
print(model_ma_5_fit.summary())
print("\nLLR test p-value = " + str(LLR_test(model_ma_4, model_ma_5)))

model_ma_6 = ARIMA(close_train_diff[1:], order = (0,0,6))
model_ma_6_fit = model_ma_6.fit()
print(model_ma_6_fit.summary())
print("\nLLR test p-value = " + str(LLR_test(model_ma_5, model_ma_6)))

model_ma_7 = ARIMA(close_train_diff[1:], order = (0,0,7))
model_ma_7_fit = model_ma_7.fit()
print(model_ma_7_fit.summary())
print("\nLLR test p-value = " + str(LLR_test(model_ma_6, model_ma_7)))

model_ma_8 = ARIMA(close_train_diff[1:], order = (0,0,8))
model_ma_8_fit = model_ma_8.fit()
print(model_ma_8_fit.summary())
print("\nLLR test p-value = " + str(LLR_test(model_ma_7, model_ma_8)))

model_ma_9 = ARIMA(close_train_diff[1:], order = (0,0,9))
model_ma_9_fit = model_ma_9.fit()
print(model_ma_9_fit.summary())
print("\nLLR test p-value = " + str(LLR_test(model_ma_1, model_ma_9, DF=8)))

model_ma_10 = ARIMA(close_train_diff[1:], order = (0,0,10))
model_ma_10_fit = model_ma_10.fit()
print(model_ma_10_fit.summary())
print("\nLLR test p-value = " + str(LLR_test(model_ma_1, model_ma_10, DF=9)))

model_ma_11 = ARIMA(close_train_diff[1:], order = (0,0,11))
model_ma_11_fit = model_ma_11.fit()
print(model_ma_11_fit.summary())
print("\nLLR test p-value = " + str(LLR_test(model_ma_10, model_ma_11, DF=1)))

model_ma_1_fit.aic
model_ma_2_fit.aic
model_ma_3_fit.aic
model_ma_4_fit.aic
model_ma_5_fit.aic
model_ma_6_fit.aic
model_ma_7_fit.aic
model_ma_8_fit.aic
model_ma_9_fit.aic
model_ma_10_fit.aic
model_ma_11_fit.aic

residui_ma = model_ma_10_fit.resid
residui_ma.mean()
residui_ma.var()
residui_ma.plot()
plt.show()
adf_test(residui_ma)


forecasts_ma = model_ma_10_fit.forecast(len(close_test_diff)) # forecast per il numero di osservazioni presenti nel test set

date_range = pd.date_range(start=close_test.index.min(), periods=len(forecasts_ma), freq='B')
forecasts_ma = pd.Series(forecasts_ma, index=date_range)
plt.plot(forecasts_ma)
plt.plot(close_test_diff)
plt.show()

# ADESSO trasformo i forecast appena effettuati in prezzi effettivi in dollari e li compariamo con il test set
forecasted_price_ma = []
start_price = close_train.Close[-1]
forecasted_price_ma.append(start_price)

for i in range(len(forecasts_ma)):
    new_price = forecasted_price_ma[-1] + ((forecasts[i])/100) * forecasted_price_ma[-1]
    forecasted_price_ma.append(new_price)

date_range = pd.date_range(start=close_test.index.min(), periods=len(forecasted_price_ma)-1, freq='B')

# Creare una serie Pandas con i prezzi e le date come indice
forecasted_price_ma = pd.Series(forecasted_price_ma[1:], index=date_range)

plt.plot(forecasted_price_ma)
close_train.Close.plot(c='b')
close_test.Close.plot(c='r')
close_test.drift_forecast.plot(c='purple')
plt.show()   

## LLR solo per modelli nested


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

# Download Apple's daily stock prices
apple = yf.download("AAPL", start="2019-01-01", end="2024-01-01")

# Calculate daily percentage return
apple['Daily_Return'] = apple['Adj Close'].pct_change() * 100

# Drop missing values
apple.dropna(inplace=True)

# Plot the original series
plt.figure(figsize=(10, 6))
plt.plot(apple['Daily_Return'], label='Daily Returns')
plt.title('Apple Daily Returns')
plt.xlabel('Date')
plt.ylabel('Daily Returns (%)')
plt.legend()
plt.show()

# Split the data into train and test sets (80/20)
train_size = int(len(apple) * 0.8)
train, test = apple['Daily_Return'][:train_size], apple['Daily_Return'][train_size:]

# Set frequency of the date index
train.index = pd.date_range(start=train.index[0], periods=len(train), freq='B')
test.index = pd.date_range(start=test.index[0], periods=len(test), freq='B')

# Fit MA model
ma_model = ARIMA(train, order=(0,0,9))
ma_fit = ma_model.fit()

# Fit AR model
ar_model = ARIMA(train, order=(10,0,0))
ar_fit = ar_model.fit()

# Fit ARMA model
arma_model = ARIMA(train, order=(8,0,7))
arma_fit = arma_model.fit()

# Forecast
ma_forecast = ma_fit.forecast(steps=len(test))
ar_forecast = ar_fit.forecast(steps=len(test))
arma_forecast = arma_fit.forecast(steps=len(test))

# Plot forecasts
plt.figure(figsize=(12, 8))
plt.plot(test.index, test.values, label='Actual')
plt.plot(test.index, ma_forecast, label='MA Forecast')
plt.plot(test.index, ar_forecast, label='AR Forecast')
plt.plot(test.index, arma_forecast, label='ARMA Forecast')
plt.title('Apple Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Daily Returns (%)')
plt.legend()
plt.show()

