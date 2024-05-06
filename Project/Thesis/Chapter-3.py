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
plt.rcParams.update({'figure.figsize':(18,8),'xtick.labelsize': 20, 'ytick.labelsize': 20})
fig, ax = plt.subplots(1,2, figsize=(18,8))
ax[0].plot(close_train, color='blue'); ax[0].set_ylabel('Prezzo $', size = 18); ax[0].set_xlabel('Data', size=18); ax[0].set_title('Serie \'close_train\'', size=19)
plot_acf(close_train,auto_ylims=True, lags=40, zero=True, c='blue', ax=ax[1]); ax[1].set_title('ACF Plot - \'close_train\'', size=18)
plt.ylabel('Coefficiente di autocorrelazione', size=18)
plt.xlabel('Lags', size=18)
plt.xticks(size=16)
plt.yticks(size=16)
ax[0].tick_params(axis='x', labelsize=16, rotation=30)
ax[0].tick_params(axis='y', labelsize=16)
plt.show() 

# 2) seasonal decompose per mostrare che comunque non c'è stagionalità:
seasonal_decomposition = seasonal_decompose(close_train, extrapolate_trend=1)
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
seasonal_decomposition.observed.plot(ax=axes[0], legend=False, color='blue')
axes[0].set_ylabel('Observed', fontsize=20); axes[0].yaxis.set_label_coords(-0.1, 0.5)  # Imposta il font degli ylabels
seasonal_decomposition.trend.plot(ax=axes[1], legend=False, color='blue')
axes[1].set_ylabel('Trend', fontsize=20); axes[1].yaxis.set_label_coords(-0.1, 0.5)  # Imposta il font degli ylabels
seasonal_decomposition.seasonal.plot(ax=axes[2], legend=False, color='blue')
axes[2].set_ylabel('Seasonal', fontsize=20); axes[2].yaxis.set_label_coords(-0.1, 0.5)  # Imposta il font degli ylabels
seasonal_decomposition.resid.plot(ax=axes[3], legend=False, color='blue')
axes[3].set_ylabel('Residual', fontsize=20); axes[3].yaxis.set_label_coords(-0.1, 0.5)   # Imposta il font degli ylabels
plt.tight_layout()
plt.show()
plt.show()

# 3) Differenzio la serie e faccio nuovamente il test di stazionarietà:
close_train_diff = ((close_train.diff()/close_train.shift(1))*100)[1:]
close_test_diff = ((close_test['Close'].diff()/close_test['Close'].shift(1))*100).bfill()

plt.rcdefaults()
close_train_diff.plot(c='blue', figsize=(15,7), label='close_train')
close_test_diff.plot(c='red', label='close_test')
plt.ylabel('Variazione %', size=20)
plt.xlabel('Data', size=20)
#plt.title('Rendimenti giornalieri percentuali (%)', size=20)
plt.xticks(size=18)
plt.yticks(size=18)
plt.legend(fontsize=18)
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
plot_acf(close_train_diff, ax=axes[0],auto_ylims=False, lags=40, zero=True, c='blue', title=None); axes[0].set_xlabel('Lag', size=18)
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
axes[0].set_xticklabels(np.arange(-5,41, step=5),fontsize=16)
axes[0].set_yticklabels([-0.2,0.0,0.2,0.4,0.6,0.8,1.0],fontsize=16)
# plotto a destra il pacf:
plot_pacf(close_train_diff, ax=axes[1],auto_ylims=False, lags=40, zero=True, method='ols', c='blue', title=None); axes[1].set_xlabel('Lag', size=18)
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
axes[1].set_xticklabels(np.arange(-5,41, step=5),fontsize=16)
axes[1].set_yticklabels([-0.2,0.0,0.2,0.4,0.6,0.8,1.0],fontsize=16)
# plotto a destra il pacf
# qui sistemo un po' di parametri per sistemare i grafici:
axes[0].set_ylim(-0.2, 1.05)
axes[1].set_ylim(-0.2, 1.05)
axes[0].set_title("Autocorrelation Function", fontsize=20)
axes[1].set_title("Partial Autocorrelation Function", fontsize=20)
#fig.suptitle('ACF e PACF per la serie di addestramento differenziata', fontsize=20, y=1)
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
plt.rcParams.update({'figure.figsize':(15,7),'xtick.labelsize': 16, 'ytick.labelsize': 16})
fig, ax = plt.subplots(1,2)
residui_ar_1.plot(ax=ax[0], c='blue'); ax[0].set_title('Time plot', size=18); ax[0].set_xlabel('Data', size=18); ax[0].set_ylabel('Rendimento %', size=18)
residui_ar_1.plot(kind='hist', bins=50, ax=ax[1], color='blue'); ax[1].set_title('Istogramma', size=18); ax[1].set_xlabel('Rendimento %', size=18); ax[1].set_ylabel('Frequenza', size=18)
ax[0].axhline(y=mean, color='red', linestyle='--', label=r'$\mu$', lw=2)
ax[1].axvline(x=sd, color='red', linestyle='--', label=r'+1 $\sigma$', lw=2)
ax[1].axvline(x=-sd, color='red', linestyle='--', label=r'-1 $\sigma$', lw=2)
ax[1].axvline(x=mean, color='orange', linestyle='--', label=r'$\mu$', lw=2)
ax[0].legend()
ax[1].legend()
#fig.suptitle('Analisi grafica dei residui del modello AR(1):', size=20)
plt.show()

plt.rcParams.update({'figure.figsize':(10,7),'xtick.labelsize': 16, 'ytick.labelsize': 16})
plot_acf(residui_ar_1,auto_ylims=True, lags=40, zero=False, c='blue', title=None)
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
#plt.title('ACF per i residui del modello AR(1)', size=20)
plt.show()

# Ciclo FOR per addestrare modelli AR con ordini da 1 a 20 e calcolare le metriche
aic = []
bic = []
hqic = []
sse = []

for i in range(1, 21):
    model = ARIMA(close_train_diff[1:], order=(i,0,0)).fit()
    aic.append(model.aic)
    bic.append(model.bic)
    hqic.append(model.hqic)
    sse.append(model.sse)
# Visualizzazione delle metriche
fig, ax = plt.subplots(1,2, figsize=(14, 5))
plt.subplots_adjust(hspace=.5, wspace=.3)
ax[0].plot(range(1, 21), aic, label='AIC', color='blue', lw=2); ax[0].set_xlabel('Lags', size=18); ax[0].set_title('Criteri Informativi', size=18)
ax[0].plot(range(1, 21), bic, label='BIC', color='orange', lw=2); 
ax[0].plot(range(1, 21), hqic, label='HQIC', color='green', lw=2); 
ax[1].plot(range(1, 21), sse, label='SSE', color='blue', lw=2); ax[1].set_xlabel('Lags', size=18); ax[1].set_title('Sum of Squared Errors', size=18)
ax[0].scatter(aic.index(min(aic))+1, min(aic), zorder=3, color='red', lw=4)
ax[0].scatter(bic.index(min(bic))+1, min(bic), zorder=3, color='red', lw=4)
ax[0].scatter(hqic.index(min(hqic))+1, min(hqic), zorder=3, color='red', lw=4)
ax[0].legend(fontsize=16)
#fig.suptitle('Metriche di valutazione dei modelli AR(p)', size=20, y=1.0)
ax[0].set_xticks(range(1, 21,2))
ax[1].set_xticks(range(1, 21,2))
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
residui_ar_9.plot(ax=ax[0], c='blue'); ax[0].set_title('Time plot', size=18); ax[0].set_xlabel('Data', size=18); ax[0].set_ylabel('Rendimento %', size=18)
residui_ar_9.plot(kind='hist', bins=50, ax=ax[1], color='blue'); ax[1].set_title('Istogramma', size=18); ax[1].set_xlabel('Rendimento %', size=18); ax[1].set_ylabel('Frequenza', size=18)
ax[0].axhline(y=mean, color='red', linestyle='--', label=r'$\mu$', lw=2)
ax[1].axvline(x=sd, color='red', linestyle='--', label=r'+1 $\sigma$', lw=2)
ax[1].axvline(x=-sd, color='red', linestyle='--', label=r'-1 $\sigma$', lw=2)
ax[1].axvline(x=mean, color='orange', linestyle='--', label=r'$\mu$', lw=2)
ax[0].legend()
ax[1].legend()
#fig.suptitle('Analisi grafica dei residui del modello AR(9):', size=20)
plt.show()

fig, ax = plt.subplots(1,2)
plt.rcParams.update({'figure.figsize':(15,7),'xtick.labelsize': 16, 'ytick.labelsize': 16})
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
#fig.suptitle('ACF per i residui', size=20)
ax[0].set_title('ACF Residui AR(1)', size=20)
ax[1].set_title('ACF Residui AR(9)', size=20)
ax[0].set_xlabel('Lags', size=18)
ax[1].set_xlabel('Lags', size=18)
plt.show()





# Modello MA(q)
plt.rcdefaults()
plt.rcParams.update({'figure.figsize':(14,6)})
fig, ax = plt.subplots(1, 2, sharex=False)

# Differenziazione di 1° ordine
# plotto a sinistra l'acf:
plot_acf(close_train_diff, ax=ax[0],auto_ylims=False, lags=40, zero=True, c='blue', title=None); ax[0].set_xlabel('Lag', size=18)
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
bic = []
hqic = []
for i in range(1, 21):
    model = ARIMA(close_train_diff[1:], order=(0,0,i)).fit()
    aic.append(model.aic)
    bic.append(model.bic)
    hqic.append(model.hqic)
# Visualizzazione delle metriche
ax[1].plot(range(1, 21), aic, label='AIC', color='blue', lw=2); ax[1].set_xlabel('Lag', size=18); ax[1].set_title('Akaike Information Criterion', size=18)
ax[1].plot(range(1, 21), bic, label='BIC', color='orange', lw=2); ax[1].set_xlabel('Lag', size=18); ax[1].set_title('Bayesian Information Criterion', size=18)
ax[1].plot(range(1, 21), hqic, label='HQIC', color='green', lw=2); ax[1].set_xlabel('Lag', size=18); ax[1].set_title('Information Criteria', size=18)
ax[1].scatter(aic.index(min(aic))+1, min(aic), zorder=3, color='red', lw=4)
ax[1].scatter(bic.index(min(bic))+1, min(bic), zorder=3, color='red', lw=4)
ax[1].scatter(hqic.index(min(hqic))+1, min(hqic), zorder=3, color='red', lw=4)
ax[1].set_xticks(range(1, 21),size=18)
ax[0].set_ylim(-0.2, 1.05)
ax[0].set_title("ACF plot", fontsize=18)
ax[1].tick_params(axis='x', labelsize=16)  # Cambia la dimensione dei ticklabels sull'asse x
ax[1].tick_params(axis='y', labelsize=16)  # Cambia la dimensione dei ticklabels sull'asse y
ax[0].tick_params(axis='x', labelsize=16)  # Cambia la dimensione dei ticklabels sull'asse x
ax[0].tick_params(axis='y', labelsize=16)  # Cambia la dimensione dei ticklabels sull'asse y
plt.legend(fontsize=15)
plt.show()


# Fittiamo i modelli MA(1), MA(10), MA(14)
#modello 1
model_ma_1 = ARIMA(close_train_diff[1:], order=(0,0,1))
model_ma_1_fit = model_ma_1.fit()
model_ma_1_fit.summary()

residui_ma_1 = model_ma_1_fit.resid
var = residui_ma_1.var()
sd = np.sqrt(var)
mean = residui_ma_1.mean()

plt.rcParams.update({'figure.figsize':(15,7)})
fig, ax = plt.subplots(1,2)
residui_ma_1.plot(ax=ax[0], c='blue'); ax[0].set_title('Time plot', size=18); ax[0].set_xlabel('Data', size=16); ax[0].set_ylabel('Rendimento %', size=16)
residui_ma_1.plot(kind='hist', bins=50, ax=ax[1], color='blue'); ax[1].set_title('Istogramma', size=18); ax[1].set_xlabel('Rendimento %', size=16); ax[1].set_ylabel('Frequenza', size=16)
ax[0].axhline(y=mean, color='red', linestyle='--', label=r'$\mu$', lw=2)
ax[1].axvline(x=sd, color='red', linestyle='--', label=r'+1 $\sigma$', lw=2)
ax[1].axvline(x=-sd, color='red', linestyle='--', label=r'-1 $\sigma$', lw=2)
ax[1].axvline(x=mean, color='orange', linestyle='--', label=r'$\mu$', lw=2)
ax[0].legend()
ax[1].legend()
fig.suptitle('Analisi grafica dei residui del modello MA(1):', size=20)
plt.show()

plt.rcParams.update({'figure.figsize':(10,7)})
plot_acf(residui_ma_1,auto_ylims=True, lags=40, zero=False, c='blue')
plt.xlabel('Lags', size=16)
plt.title('ACF dei residui del modello MA(14)', size=20)
plt.show()

# modello 10
model_ma_10 = ARIMA(close_train_diff[1:], order=(0,0,10))
model_ma_10_fit = model_ma_10.fit()
model_ma_10_fit.summary()

residui_ma_10 = model_ma_10_fit.resid
var = residui_ma_10.var()
sd = np.sqrt(var)
mean = residui_ma_10.mean()

plt.rcParams.update({'figure.figsize':(15,7)})
fig, ax = plt.subplots(1,2)
residui_ma_10.plot(ax=ax[0], c='blue'); ax[0].set_title('Time plot', size=18); ax[0].set_xlabel('Data', size=16); ax[0].set_ylabel('Rendimento %', size=16)
residui_ma_10.plot(kind='hist', bins=50, ax=ax[1], color='blue'); ax[1].set_title('Istogramma', size=18); ax[1].set_xlabel('Rendimento %', size=16); ax[1].set_ylabel('Frequenza', size=16)
ax[0].axhline(y=mean, color='red', linestyle='--', label=r'$\mu$', lw=2)
ax[1].axvline(x=sd, color='red', linestyle='--', label=r'+1 $\sigma$', lw=2)
ax[1].axvline(x=-sd, color='red', linestyle='--', label=r'-1 $\sigma$', lw=2)
ax[1].axvline(x=mean, color='orange', linestyle='--', label=r'$\mu$', lw=2)
ax[0].legend()
ax[1].legend()
fig.suptitle('Analisi grafica dei residui del modello MA(10):', size=20)
plt.show()

plt.rcParams.update({'figure.figsize':(10,7)})
plot_acf(residui_ma_10,auto_ylims=True, lags=40, zero=False, c='blue')
plt.xlabel('Lags', size=16)
plt.title('ACF dei residui del modello MA(10)', size=20)
plt.show()


#modello 14
model_ma_14 = ARIMA(close_train_diff[1:], order=(0,0,14))
model_ma_14_fit = model_ma_14.fit()
model_ma_14_fit.summary()


residui_ma_14 = model_ma_14_fit.resid
var = residui_ma_14.var()
sd = np.sqrt(var)
mean = residui_ma_14.mean()

plt.rcParams.update({'figure.figsize':(15,7),'xtick.labelsize': 18, 'ytick.labelsize': 18})
fig, ax = plt.subplots(1,2)
residui_ma_14.plot(ax=ax[0], c='blue'); ax[0].set_title('Time plot', size=20); ax[0].set_xlabel('Data', size=18); ax[0].set_ylabel('Rendimento %', size=18)
residui_ma_14.plot(kind='hist', bins=50, ax=ax[1], color='blue'); ax[1].set_title('Istogramma', size=20); ax[1].set_xlabel('Rendimento %', size=18); ax[1].set_ylabel('Frequenza', size=18)
ax[0].axhline(y=mean, color='red', linestyle='--', label=r'$\mu$', lw=2)
ax[1].axvline(x=sd, color='red', linestyle='--', label=r'+1 $\sigma$', lw=2)
ax[1].axvline(x=-sd, color='red', linestyle='--', label=r'-1 $\sigma$', lw=2)
ax[1].axvline(x=mean, color='orange', linestyle='--', label=r'$\mu$', lw=2)
ax[0].legend(fontsize=16)
ax[1].legend(fontsize=16)

#fig.suptitle('Analisi grafica dei residui del modello MA(14):', size=20)
plt.show()

plt.rcParams.update({'figure.figsize':(10,7)})
plot_acf(residui_ma_14,auto_ylims=True, lags=40, zero=False,title=None, c='blue')
plt.xlabel('Lags', size=18)
plt.xticks(size=16)
plt.yticks(size=16)
#plt.title('ACF dei residui del modello MA(14)', size=20)
plt.show()



# ARMA (p,q) Model:
import seaborn as sns

# I tuoi dati in formato pd.Series
# Assicurati di sostituire questo con la tua serie di dati

# Genera le combinazioni di p e q
max_p = 10
max_q = 10
pq_combinations = [(p, q) for p in range(max_p + 1) for q in range(max_q + 1)]

# Inizializza una matrice per salvare i valori di AIC
aic_matrix = np.zeros((max_p + 1, max_q + 1))
bic_matrix = np.zeros((max_p + 1, max_q + 1))
hqic_matrix = np.zeros((max_p + 1, max_q + 1))

# Addestra i modelli ARMA e salva i valori AIC
for p, q in pq_combinations:
    if p == 0 and q == 0:
        aic_matrix[p, q] = np.nan
        continue  # Evita ARMA(0,0)
    try:
        arma_model = ARIMA(close_train_diff, order=(p,0,q)).fit()
        aic_matrix[p, q] = arma_model.aic
        bic_matrix[p, q] = arma_model.bic
        hqic_matrix[p, q] = arma_model.hqic
    except Exception as e:
        print(f"Errore durante l'addestramento del modello ARMA({p},{q}): {str(e)}")
        aic_matrix[p, q] = np.nan
        bic_matrix[p, q] = np.nan
        hqic_matrix[p, q] = np.nan  # Imposta a NaN se il modello non converge

# Crea un DataFrame per i risultati
aic_df = pd.DataFrame(aic_matrix, index=range(max_p + 1), columns=range(max_q + 1))
bic_df = pd.DataFrame(bic_matrix, index=range(max_p + 1), columns=range(max_q + 1))
hqic_df = pd.DataFrame(hqic_matrix, index=range(max_p + 1), columns=range(max_q + 1))

hqic_df[0][0] = np.nan
bic_df[0][0] = np.nan
# Trova i valori minimi di AIC e le relative coordinate
min_aic = aic_df.stack().min()
min_aic_idx = np.unravel_index(np.nanargmin(aic_matrix), aic_matrix.shape)
min_bic = bic_df.stack().min()
min_bic_idx = np.unravel_index(np.nanargmin(bic_matrix), bic_matrix.shape)
min_hqic = hqic_df.stack().min()
min_hqic_idx = np.unravel_index(np.nanargmin(hqic_matrix), hqic_matrix.shape)

# Visualizza la heatmap  AIC con Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(aic_df, annot=True, cmap='viridis', fmt='.1f', cbar=True)
plt.title('Heatmap dei valori AIC per i modelli ARMA(p,q)', size=18, pad=15)
plt.xlabel('q', size=16, labelpad=15)
plt.ylabel('p', size=16, labelpad=15)
#plt.scatter(min_aic_idx[1], min_aic_idx[0], marker='X', color='red', s=100, label=f'Min AIC: {min_aic}')
rect = plt.Rectangle((min_aic_idx[1], min_aic_idx[0]), 1, 1, fill=False, edgecolor='orangered', linewidth=2)
plt.gca().add_patch(rect)
plt.show()

# Visualizza la heatmap  BIC con Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(bic_df, annot=True, cmap='viridis', fmt='.1f', cbar=True)
plt.title('Heatmap dei valori BIC per i modelli ARMA(p,q)', size=18, pad=15)
plt.xlabel('q', size=16, labelpad=15)
plt.ylabel('p', size=16, labelpad=15)
#plt.scatter(min_aic_idx[1], min_aic_idx[0], marker='X', color='red', s=100, label=f'Min AIC: {min_aic}')
rect = plt.Rectangle((min_bic_idx[1], min_bic_idx[0]), 1, 1, fill=False, edgecolor='orangered', linewidth=2)
plt.gca().add_patch(rect)
plt.show()

# Visualizza la heatmap  HQIC con Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(hqic_df, annot=True, cmap='viridis', fmt='.1f', cbar=True)
plt.title('Heatmap dei valori HQIC per i modelli ARMA(p,q)', size=18, pad=15)
plt.xlabel('q', size=16, labelpad=15)
plt.ylabel('p', size=16, labelpad=15)
#plt.scatter(min_aic_idx[1], min_aic_idx[0], marker='X', color='red', s=100, label=f'Min AIC: {min_aic}')
rect = plt.Rectangle((min_hqic_idx[1], min_hqic_idx[0]), 1, 1, fill=False, edgecolor='orangered', linewidth=2)
plt.gca().add_patch(rect)
plt.show()





# modello migliore è ARMA(3,7)
model_ar_3_ma_7 = ARIMA(close_train_diff, order=(3,0,7))
model_ar_3_ma_7_fit = model_ar_3_ma_7.fit()
model_ar_3_ma_7_fit.summary()

residui_ar_3_ma_7 = model_ar_3_ma_7_fit.resid
var = residui_ar_3_ma_7.var()
sd = np.sqrt(var)
mean = residui_ar_3_ma_7.mean()

plt.rcParams.update({'figure.figsize':(15,7)})
fig, ax = plt.subplots(1,2)
residui_ar_3_ma_7.plot(ax=ax[0], c='blue'); ax[0].set_title('Time plot', size=20); ax[0].set_xlabel('Data', size=18); ax[0].set_ylabel('Rendimento %', size=18)
residui_ar_3_ma_7.plot(kind='hist', bins=50, ax=ax[1], color='blue'); ax[1].set_title('Istogramma', size=20); ax[1].set_xlabel('Rendimento %', size=18); ax[1].set_ylabel('Frequenza', size=18)
ax[0].axhline(y=mean, color='red', linestyle='--', label=r'$\mu$', lw=2)
ax[1].axvline(x=sd, color='red', linestyle='--', label=r'+1 $\sigma$', lw=2)
ax[1].axvline(x=-sd, color='red', linestyle='--', label=r'-1 $\sigma$', lw=2)
ax[1].axvline(x=mean, color='orange', linestyle='--', label=r'$\mu$', lw=2)
ax[0].legend()
ax[1].legend()
#fig.suptitle('Analisi grafica dei residui del modello ARMA(3,7):', size=20)
plt.show()

plt.rcParams.update({'figure.figsize':(10,7)})
plot_acf(residui_ar_3_ma_7,auto_ylims=True, lags=40, zero=False, c='blue',title=None)
plt.xlabel('Lag', size=16)
#plt.title('ACF dei residui del modello ARMA(3,7)', size=20)
plt.show()

# modello migliore bic ARMA(2,2)
model_ar_2_ma_2 = ARIMA(close_train_diff, order=(2,0,2))
model_ar_2_ma_2_fit = model_ar_2_ma_2.fit()
model_ar_2_ma_2_fit.summary()

residui_ar_2_ma_2 = model_ar_2_ma_2_fit.resid
var = residui_ar_2_ma_2.var()
sd = np.sqrt(var)
mean = residui_ar_2_ma_2.mean()

plt.rcParams.update({'figure.figsize':(15,7)})
fig, ax = plt.subplots(1,2)
residui_ar_2_ma_2.plot(ax=ax[0], c='blue'); ax[0].set_title('Time plot', size=18); ax[0].set_xlabel('Data', size=16); ax[0].set_ylabel('Rendimento %', size=16)
residui_ar_2_ma_2.plot(kind='hist', bins=50, ax=ax[1], color='blue'); ax[1].set_title('Istogramma', size=18); ax[1].set_xlabel('Rendimento %', size=16); ax[1].set_ylabel('Frequenza', size=16)
ax[0].axhline(y=mean, color='red', linestyle='--', label=r'$\mu$', lw=2)
ax[1].axvline(x=sd, color='red', linestyle='--', label=r'+1 $\sigma$', lw=2)
ax[1].axvline(x=-sd, color='red', linestyle='--', label=r'-1 $\sigma$', lw=2)
ax[1].axvline(x=mean, color='orange', linestyle='--', label=r'$\mu$', lw=2)
ax[0].legend()
ax[1].legend()
fig.suptitle('Analisi grafica dei residui del modello ARMA(2,2):', size=20)
plt.show()

plt.rcParams.update({'figure.figsize':(10,7)})
plot_acf(residui_ar_2_ma_2,auto_ylims=True, lags=40, zero=False, c='blue',title=None)
plt.xlabel('Lag', size=18)
#plt.title('ACF dei residui del modello ARMA(2,2)', size=20)
plt.show()

# modello migliore bic ARMA(2,2)
model_ar_2_ma_5 = ARIMA(close_train_diff, order=(2,0,5))
model_ar_2_ma_5_fit = model_ar_2_ma_5.fit()
model_ar_2_ma_5_fit.summary()

residui_ar_2_ma_5 = model_ar_2_ma_5_fit.resid
var = residui_ar_2_ma_5.var()
sd = np.sqrt(var)
mean = residui_ar_2_ma_5.mean()

plt.rcParams.update({'figure.figsize':(15,7)})
fig, ax = plt.subplots(1,2)
residui_ar_2_ma_5.plot(ax=ax[0], c='blue'); ax[0].set_title('Time plot', size=18); ax[0].set_xlabel('Data', size=16); ax[0].set_ylabel('Rendimento %', size=16)
residui_ar_2_ma_5.plot(kind='hist', bins=50, ax=ax[1], color='blue'); ax[1].set_title('Istogramma', size=18); ax[1].set_xlabel('Rendimento %', size=16); ax[1].set_ylabel('Frequenza', size=16)
ax[0].axhline(y=mean, color='red', linestyle='--', label=r'$\mu$', lw=2)
ax[1].axvline(x=sd, color='red', linestyle='--', label=r'+1 $\sigma$', lw=2)
ax[1].axvline(x=-sd, color='red', linestyle='--', label=r'-1 $\sigma$', lw=2)
ax[1].axvline(x=mean, color='orange', linestyle='--', label=r'$\mu$', lw=2)
ax[0].legend()
ax[1].legend()
fig.suptitle('Analisi grafica dei residui del modello ARMA(2,5):', size=20)
plt.show()

plt.rcParams.update({'figure.figsize':(10,7)})
plot_acf(residui_ar_2_ma_5,auto_ylims=True, lags=40, zero=False, c='blue')
plt.xlabel('Lag', size=16)
plt.title('ACF dei residui del modello ARMA(2,5)', size=20)
plt.show()


# Si producono i forecast con ciascun modello:
forecast_ar_1 = model_ar_1_fit.forecast(len(close_test))
forecast_ar_9 = model_ar_9_fit.forecast(len(close_test))
forecast_ma_1 = model_ma_1_fit.forecast(len(close_test))
forecast_ma_10 = model_ma_10_fit.forecast(len(close_test))
forecast_ma_14 = model_ma_14_fit.forecast(len(close_test))
forecast_ar_2_ma_2 = model_ar_2_ma_2_fit.forecast(len(close_test))
forecast_ar_2_ma_5 = model_ar_2_ma_5_fit.forecast(len(close_test))
forecast_ar_3_ma_7 = model_ar_3_ma_7_fit.forecast(len(close_test))

# Si estrae l'ultimo prezzo disponibile nel set di addestramento:
ultimo_prezzo = close_train['Close'].iloc[-1]

# Si trasformano i forecast (rendimenti percentuali giornalieri) in prezzi:
price_forecast_ar_1 = ultimo_prezzo * (1 + forecast_ar_1 / 100).cumprod()
price_forecast_ar_9 = ultimo_prezzo * (1 + forecast_ar_9 / 100).cumprod()
price_forecast_ma_1 = ultimo_prezzo * (1 + forecast_ma_1 / 100).cumprod()
price_forecast_ma_10 = ultimo_prezzo * (1 + forecast_ma_10 / 100).cumprod()
price_forecast_ma_14 = ultimo_prezzo * (1 + forecast_ma_14 / 100).cumprod()
price_forecast_ar_2_ma_2 = ultimo_prezzo * (1 + forecast_ar_2_ma_2 / 100).cumprod()
price_forecast_ar_2_ma_5 = ultimo_prezzo * (1 + forecast_ar_2_ma_5 / 100).cumprod()
price_forecast_ar_3_ma_7 = ultimo_prezzo * (1 + forecast_ar_3_ma_7 / 100).cumprod()

import matplotlib.patheffects as pe
plt.rcParams['figure.dpi'] = 150
plt.rcParams.update({'figure.figsize':(15,7)})
close_train.Close[-25:].plot(label='Train set [-25:]', color='blue', style='--', alpha=0.5, lw=2)
close_test.Close.plot(label='Test set', color='blue', lw=2)
close_test.drift_forecast.plot(label='Forecasts Drift', color='darkviolet', lw=2)
price_forecast_ar_1.plot(label='Forecasts AR(1)', color='black', zorder=3, style=':', lw=3)
price_forecast_ar_9.plot(label='Forecasts AR(9)', color='#faa307',style=':', lw=3)
price_forecast_ma_1.plot(label='Forecasts MA(1)', color='#f72585', lw=3)
price_forecast_ma_10.plot(label='Forecasts MA(10)', color='#70e000', zorder=2)
price_forecast_ma_14.plot(label='Forecasts MA(14)')
price_forecast_ar_2_ma_2.plot(label='Forecasts ARMA(2,2)')
price_forecast_ar_2_ma_5.plot(label='Forecasts ARMA(2,5)')
price_forecast_ar_3_ma_7.plot(label='Forecasts ARMA(3,7)')
plt.ylabel('Prezzi ($)', size=16)
plt.xlabel('Data', size=16)
#plt.title('Forecasts per il periodo di test', size=20, pad=15)
plt.legend(fontsize=12)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

# Calcolo il mean square error delle previsioni 
close_test_copy = close_test.copy()
close_test_copy['ar_1'] = price_forecast_ar_1
close_test_copy['ar_9'] = price_forecast_ar_9
close_test_copy['ma_1'] = price_forecast_ma_1
close_test_copy['ma_10'] = price_forecast_ma_10
close_test_copy['ma_14'] = price_forecast_ma_14
close_test_copy['ar_2_ma_2'] = price_forecast_ar_2_ma_2
close_test_copy['ar_2_ma_5'] = price_forecast_ar_2_ma_5
close_test_copy['ar_3_ma_7'] = price_forecast_ar_3_ma_7

from sklearn.metrics import mean_squared_error
drift_mse = round(mean_squared_error(close_test_copy.Close, close_test_copy.drift_forecast),2)
ar1_mse = round(mean_squared_error(close_test_copy.Close, close_test_copy.ar_1),2)
ar9_mse = round(mean_squared_error(close_test_copy.Close, close_test_copy.ar_9),2)

ma1_mse = round(mean_squared_error(close_test_copy.Close, close_test_copy.ma_1),2)
ma10_mse = round(mean_squared_error(close_test_copy.Close, close_test_copy.ma_10),2)
ma14_mse = round(mean_squared_error(close_test_copy.Close, close_test_copy.ma_14),2)

arma22_mse = round(mean_squared_error(close_test_copy.Close, close_test_copy.ar_2_ma_2),2)
arma25_mse = round(mean_squared_error(close_test_copy.Close, close_test_copy.ar_2_ma_5),2)
arma37_mse = round(mean_squared_error(close_test_copy.Close, close_test_copy.ar_3_ma_7),2)

benchmarks_errors = pd.DataFrame({'MSE': [drift_mse, ar1_mse, ar9_mse, ma1_mse, ma10_mse, ma14_mse, arma22_mse, arma25_mse, arma37_mse]}, 
                                 index = ['Drift Forecast','AR(1)','AR(9)','MA(1)','MA(10)','MA(14)','ARMA(2,2)','ARMA(2,5)','ARMA(3,7)'])
benchmarks_errors






# rolling window forecast: 
close_test_diff_copy = close_test_diff.copy()
close_train_diff_copy = close_train_diff.copy()['Close']

window_size = 1
iterations = int(len(close_test)/window_size)
forecast = pd.Series()
start = 0

for i in range(iterations):
    model = ARIMA(close_train_diff_copy, order=(3,0,7)).fit()
    forecast = forecast._append(model.forecast(steps=window_size))
    close_train_diff_copy = close_train_diff_copy._append(close_test_diff_copy[start:start+window_size])
    close_train_diff_copy = close_train_diff_copy[window_size:]
    print(len(close_train_diff_copy))
    start += window_size

price_forecast_rolling = ultimo_prezzo * (1 + forecast / 100).cumprod()

price_forecast_rolling.plot(label='Rolling Predictions')
close_test.Close.plot(label='Close Test')
plt.legend()
plt.show()


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!           COME PLOTTARE LE PREVISIONI CON TANTO DI INTERVALLO DI CONFIDENZA            !!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Plot delle previsioni e degli intervalli di confidenza
fig, ax = plt.subplots(figsize=(10, 6))

# Plotta i dati osservati (se disponibili)
ax.plot(close_train_diff, label='Dati osservati', color='blue')

# Plotta le previsioni
forecast = model_ar_9_fit.forecast(steps=len(close_test))  # Numero di passi previsti nel futuro
ax.plot(forecast, label='Previsioni', color='red')

# Plotta gli intervalli di confidenza
forecast_ci = model_ar_1_fit.get_forecast(steps=len(close_test)).conf_int()  # Calcola gli intervalli di confidenza
ax.fill_between(x=forecast_ci.index, y1=forecast_ci.iloc[:,0], y2=forecast_ci.iloc[:,1], color='lightgrey')

# Aggiungi etichette, legenda, titolo, ecc.
ax.set_xlabel('Tempo')
ax.set_ylabel('Valore')
ax.legend()
plt.title('Previsioni AR(1) con intervalli di confidenza')

# Mostra il grafico
plt.show()

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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
