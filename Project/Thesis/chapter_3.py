# Import the relevant packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.tsa.stattools as sts #per l'.adfuller()
from statsmodels.tsa.seasonal import seasonal_decompose, STL, MSTL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from scipy.stats.distributions import chi2
from statsmodels.tsa.arima.model import ARIMA 

# Import the data form the CSVs we previously saved
close_train = pd.read_csv('Project\\Thesis\\Train and test data\\close_train.csv', parse_dates=True, index_col='Date')
close_test = pd.read_csv('Project\\Thesis\\Train and test data\\close_test.csv', parse_dates=True, index_col='Date')[['Close', 'drift_forecast']]

close_train.head()
close_test.head()


# Set the frequency to business day
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

# Improve the output of the ADF test
def adf_test(column):
    adf_result = sts.adfuller(column)
    adf_output = pd.Series(adf_result[0:4], index=['Test Statistic','P-value','Lags Used','Number of Observations Used'])
    for key, value in adf_result[4].items():
        adf_output[f'Critical Value ({key})'] = value
    return adf_output

adf_test(close_train)

############################################################################################################################

# Chapter 3: Modelling

# 1) ACF for the non-stationary series and adf test:
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

# 2) seasonal decompose:
seasonal_decomposition = seasonal_decompose(close_train, extrapolate_trend=1)
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
seasonal_decomposition.observed.plot(ax=axes[0], legend=False, color='blue')
axes[0].set_ylabel('Observed', fontsize=20); axes[0].yaxis.set_label_coords(-0.1, 0.5)  
seasonal_decomposition.trend.plot(ax=axes[1], legend=False, color='blue')
axes[1].set_ylabel('Trend', fontsize=20); axes[1].yaxis.set_label_coords(-0.1, 0.5)  
seasonal_decomposition.seasonal.plot(ax=axes[2], legend=False, color='blue')
axes[2].set_ylabel('Seasonal', fontsize=20); axes[2].yaxis.set_label_coords(-0.1, 0.5)  
seasonal_decomposition.resid.plot(ax=axes[3], legend=False, color='blue')
axes[3].set_ylabel('Residual', fontsize=20); axes[3].yaxis.set_label_coords(-0.1, 0.5)   
plt.tight_layout()
plt.show()
plt.show()

# 3) Differenciate the series and perform another ADF test
close_train_diff = ((close_train.diff()/close_train.shift(1))*100)[1:]
close_test_diff = ((close_test['Close'].diff()/close_test['Close'].shift(1))*100).bfill()

plt.rcdefaults()
close_train_diff.plot(c='blue', figsize=(15,7), label='close_train')
close_test_diff.plot(c='red', label='close_test')
plt.ylabel('Variazione %', size=20)
plt.xlabel('Data', size=20)
plt.xticks(size=18)
plt.yticks(size=18)
plt.legend(fontsize=18)
plt.show()

adf_test(close_train_diff)

# From the differenced series we go back to the orgiginal one just to make sure we're doing everything as we should
close_int = close_train.Close[0] + ((close_train_diff/100)*close_train.shift(1)).cumsum()
close_int.Close[0] = close_train.Close[0]
close_int; close_train

# Plot them to visually see if they match
plt.plot(close_int.index, close_int.Close, color='red', label='Serie modificata')
plt.plot(close_int.index, close_train.Close, color='blue', linestyle=':', label='Serie originale')
plt.title(f'Prezzi di chiusura giornalieri AAPL dal: {close_train.index.date.min()} al {close_train.index.date.max()}')
plt.ylabel('Prezzi $')
plt.xlabel('Data')
plt.legend()
plt.show()

# 3.1) Autoregressive model,  AR(p):
# ACF and PACF of the differences series:
plt.rcdefaults()
plt.rcParams.update({'figure.figsize':(14,6)})
fig, axes = plt.subplots(1, 2, sharex=False)

# 1st order differencing
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

axes[0].set_ylim(-0.2, 1.05)
axes[1].set_ylim(-0.2, 1.05)
axes[0].set_title("Autocorrelation Function", fontsize=20)
axes[1].set_title("Partial Autocorrelation Function", fontsize=20)
plt.show()

# Choose the AR(p) order based on the lowest AIC value
# model AR 1
model_ar_1 = ARIMA(close_train_diff[1:],order=(1,0,0))
model_ar_1_fit = model_ar_1.fit()
model_ar_1_fit.summary()

residui_ar_1 = model_ar_1_fit.resid
var = residui_ar_1.var()
sd = np.sqrt(var)
mean = residui_ar_1.mean()

# Visual analysis of the residuals 
plt.rcParams.update({'figure.figsize':(15,7),'xtick.labelsize': 16, 'ytick.labelsize': 16})
fig, ax = plt.subplots(1,2)
residui_ar_1.plot(ax=ax[0], c='blue'); ax[0].set_title('Time plot', size=18); ax[0].set_xlabel('Date', size=18); ax[0].set_ylabel('Return %', size=18)
residui_ar_1.plot(kind='hist', bins=50, ax=ax[1], color='blue'); ax[1].set_title('Histogram', size=18); ax[1].set_xlabel('Return %', size=18); ax[1].set_ylabel('Frequency', size=18)
ax[0].axhline(y=mean, color='red', linestyle='--', label=r'$\mu$', lw=2)
ax[1].axvline(x=sd, color='red', linestyle='--', label=r'+1 $\sigma$', lw=2)
ax[1].axvline(x=-sd, color='red', linestyle='--', label=r'-1 $\sigma$', lw=2)
ax[1].axvline(x=mean, color='orange', linestyle='--', label=r'$\mu$', lw=2)
ax[0].legend()
ax[1].legend()
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
plt.show()

# FOR CYCLE to train AR models with p from 1 to 20 and calculate the Information criteria
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
ax[0].plot(range(1, 21), aic, label='AIC', color='blue', lw=2); ax[0].set_xlabel('Lags', size=18); ax[0].set_title('Information Criteria', size=18)
ax[0].plot(range(1, 21), bic, label='BIC', color='orange', lw=2); 
ax[0].plot(range(1, 21), hqic, label='HQIC', color='green', lw=2); 
ax[1].plot(range(1, 21), sse, label='SSE', color='blue', lw=2); ax[1].set_xlabel('Lags', size=18); ax[1].set_title('Sum of Squared Errors', size=18)
ax[0].scatter(aic.index(min(aic))+1, min(aic), zorder=3, color='red', lw=4)
ax[0].scatter(bic.index(min(bic))+1, min(bic), zorder=3, color='red', lw=4)
ax[0].scatter(hqic.index(min(hqic))+1, min(hqic), zorder=3, color='red', lw=4)
ax[0].legend(fontsize=16)
ax[0].set_xticks(range(1, 21,2))
ax[1].set_xticks(range(1, 21,2))
plt.show()

# Fount the lowest p for the AID we prodceed with the fitting and residuals analysis
model_ar_9 = ARIMA(close_train_diff[1:], order=(9,0,0))
model_ar_9_fit = model_ar_9.fit()
model_ar_9_fit.summary()

residui_ar_9 = model_ar_9_fit.resid
var = residui_ar_9.var()
sd = np.sqrt(var)
mean = residui_ar_9.mean()

plt.rcParams.update({'figure.figsize':(15,7)})
fig, ax = plt.subplots(1,2)
residui_ar_9.plot(ax=ax[0], c='blue'); ax[0].set_title('Time plot', size=18); ax[0].set_xlabel('Date', size=18); ax[0].set_ylabel('Return %', size=18)
residui_ar_9.plot(kind='hist', bins=50, ax=ax[1], color='blue'); ax[1].set_title('Histogram', size=18); ax[1].set_xlabel('Return %', size=18); ax[1].set_ylabel('Frequency', size=18)
ax[0].axhline(y=mean, color='red', linestyle='--', label=r'$\mu$', lw=2)
ax[1].axvline(x=sd, color='red', linestyle='--', label=r'+1 $\sigma$', lw=2)
ax[1].axvline(x=-sd, color='red', linestyle='--', label=r'-1 $\sigma$', lw=2)
ax[1].axvline(x=mean, color='orange', linestyle='--', label=r'$\mu$', lw=2)
ax[0].legend()
ax[1].legend()
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
ax[0].set_title('ACF residuals AR(1)', size=20)
ax[1].set_title('ACF residuals AR(9)', size=20)
ax[0].set_xlabel('Lags', size=18)
ax[1].set_xlabel('Lags', size=18)
plt.show()



# MA(q) model
plt.rcdefaults()
plt.rcParams.update({'figure.figsize':(14,6)})
fig, ax = plt.subplots(1, 2, sharex=False)

# 1st order differencing
# plotof the ACF on the left
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

# Plot the information criteria on the right
aic = []
bic = []
hqic = []
for i in range(1, 21):
    model = ARIMA(close_train_diff[1:], order=(0,0,i)).fit()
    aic.append(model.aic)
    bic.append(model.bic)
    hqic.append(model.hqic)

ax[1].plot(range(1, 21), aic, label='AIC', color='blue', lw=2); ax[1].set_xlabel('Lag', size=18); ax[1].set_title('Akaike Information Criterion', size=18)
ax[1].plot(range(1, 21), bic, label='BIC', color='orange', lw=2); ax[1].set_xlabel('Lag', size=18); ax[1].set_title('Bayesian Information Criterion', size=18)
ax[1].plot(range(1, 21), hqic, label='HQIC', color='green', lw=2); ax[1].set_xlabel('Lag', size=18); ax[1].set_title('Information Criteria', size=18)
ax[1].scatter(aic.index(min(aic))+1, min(aic), zorder=3, color='red', lw=4); ax[1].scatter(bic.index(min(bic))+1, min(bic), zorder=3, color='red', lw=4)
ax[1].scatter(hqic.index(min(hqic))+1, min(hqic), zorder=3, color='red', lw=4); ax[1].set_xticks(range(1, 21),size=18)
ax[0].set_ylim(-0.2, 1.05); ax[0].set_title("ACF plot", fontsize=18)
ax[1].tick_params(axis='x', labelsize=16); ax[1].tick_params(axis='y', labelsize=16); ax[0].tick_params(axis='x', labelsize=16);  ax[0].tick_params(axis='y', labelsize=16)  
plt.legend(fontsize=15)
plt.show()


# Fit the models MA(1), MA(10), MA(14)
# MA(1)
model_ma_1 = ARIMA(close_train_diff[1:], order=(0,0,1))
model_ma_1_fit = model_ma_1.fit()
model_ma_1_fit.summary()

residui_ma_1 = model_ma_1_fit.resid
var = residui_ma_1.var()
sd = np.sqrt(var)
mean = residui_ma_1.mean()

plt.rcParams.update({'figure.figsize':(15,7)})
fig, ax = plt.subplots(1,2)
residui_ma_1.plot(ax=ax[0], c='blue'); ax[0].set_title('Time plot', size=18); ax[0].set_xlabel('Date', size=16); ax[0].set_ylabel('Return %', size=16)
residui_ma_1.plot(kind='hist', bins=50, ax=ax[1], color='blue'); ax[1].set_title('Histogram', size=18); ax[1].set_xlabel('Return %', size=16); ax[1].set_ylabel('Frequency', size=16)
ax[0].axhline(y=mean, color='red', linestyle='--', label=r'$\mu$', lw=2)
ax[1].axvline(x=sd, color='red', linestyle='--', label=r'+1 $\sigma$', lw=2)
ax[1].axvline(x=-sd, color='red', linestyle='--', label=r'-1 $\sigma$', lw=2)
ax[1].axvline(x=mean, color='orange', linestyle='--', label=r'$\mu$', lw=2)
ax[0].legend()
ax[1].legend()
fig.suptitle('Graphical analysis of the MA(1) residuals', size=20)
plt.show()

plt.rcParams.update({'figure.figsize':(10,7)})
plot_acf(residui_ma_1,auto_ylims=True, lags=40, zero=False, c='blue')
plt.xlabel('Lags', size=16)
plt.title('ACF of the MA(1) residuals', size=20)
plt.show()

# MA(10)
model_ma_10 = ARIMA(close_train_diff[1:], order=(0,0,10))
model_ma_10_fit = model_ma_10.fit()
model_ma_10_fit.summary()

residui_ma_10 = model_ma_10_fit.resid
var = residui_ma_10.var()
sd = np.sqrt(var)
mean = residui_ma_10.mean()

plt.rcParams.update({'figure.figsize':(15,7)})
fig, ax = plt.subplots(1,2)
residui_ma_10.plot(ax=ax[0], c='blue'); ax[0].set_title('Time plot', size=18); ax[0].set_xlabel('Date', size=16); ax[0].set_ylabel('Return %', size=16)
residui_ma_10.plot(kind='hist', bins=50, ax=ax[1], color='blue'); ax[1].set_title('Histogram', size=18); ax[1].set_xlabel('Return %', size=16); ax[1].set_ylabel('Frequency', size=16)
ax[0].axhline(y=mean, color='red', linestyle='--', label=r'$\mu$', lw=2)
ax[1].axvline(x=sd, color='red', linestyle='--', label=r'+1 $\sigma$', lw=2)
ax[1].axvline(x=-sd, color='red', linestyle='--', label=r'-1 $\sigma$', lw=2)
ax[1].axvline(x=mean, color='orange', linestyle='--', label=r'$\mu$', lw=2)
ax[0].legend()
ax[1].legend()
fig.suptitle('Graphical analysis of the MA(10) residuals:', size=20)
plt.show()

plt.rcParams.update({'figure.figsize':(10,7)})
plot_acf(residui_ma_10,auto_ylims=True, lags=40, zero=False, c='blue')
plt.xlabel('Lags', size=16)
plt.title('ACF of the MA(10) residuals', size=20)
plt.show()


# MA(14)
model_ma_14 = ARIMA(close_train_diff[1:], order=(0,0,14))
model_ma_14_fit = model_ma_14.fit()
model_ma_14_fit.summary()

residui_ma_14 = model_ma_14_fit.resid
var = residui_ma_14.var()
sd = np.sqrt(var)
mean = residui_ma_14.mean()

plt.rcParams.update({'figure.figsize':(15,7),'xtick.labelsize': 18, 'ytick.labelsize': 18})
fig, ax = plt.subplots(1,2)
residui_ma_14.plot(ax=ax[0], c='blue'); ax[0].set_title('Time plot', size=20); ax[0].set_xlabel('Data', size=18); ax[0].set_ylabel('Return %', size=18)
residui_ma_14.plot(kind='hist', bins=50, ax=ax[1], color='blue'); ax[1].set_title('Histogram', size=20); ax[1].set_xlabel('Return %', size=18); ax[1].set_ylabel('Frequency', size=18)
ax[0].axhline(y=mean, color='red', linestyle='--', label=r'$\mu$', lw=2)
ax[1].axvline(x=sd, color='red', linestyle='--', label=r'+1 $\sigma$', lw=2)
ax[1].axvline(x=-sd, color='red', linestyle='--', label=r'-1 $\sigma$', lw=2)
ax[1].axvline(x=mean, color='orange', linestyle='--', label=r'$\mu$', lw=2)
ax[0].legend(fontsize=16)
ax[1].legend(fontsize=16)
plt.show()

plt.rcParams.update({'figure.figsize':(10,7)})
plot_acf(residui_ma_14,auto_ylims=True, lags=40, zero=False,title=None, c='blue')
plt.xlabel('Lags', size=18)
plt.xticks(size=16)
plt.yticks(size=16)
plt.show()


# ARMA (p,q) Model:
import seaborn as sns
# Generate combinations of p and q
max_p = 10
max_q = 10
pq_combinations = [(p, q) for p in range(max_p + 1) for q in range(max_q + 1)]

# Initialize a matrix to save the information criteria values 
aic_matrix = np.zeros((max_p + 1, max_q + 1))
bic_matrix = np.zeros((max_p + 1, max_q + 1))
hqic_matrix = np.zeros((max_p + 1, max_q + 1))

# Train the ARMA models and saves the AIC/BIC/HQCI values
for p, q in pq_combinations:
    if p == 0 and q == 0:
        aic_matrix[p, q] = np.nan
        continue 
    try:
        arma_model = ARIMA(close_train_diff, order=(p,0,q)).fit()
        aic_matrix[p, q] = arma_model.aic
        bic_matrix[p, q] = arma_model.bic
        hqic_matrix[p, q] = arma_model.hqic
    except Exception as e:
        print(f"Error during the training of the model ARMA({p},{q}): {str(e)}")
        aic_matrix[p, q] = np.nan
        bic_matrix[p, q] = np.nan
        hqic_matrix[p, q] = np.nan  # Set the value as NaN if the model does not converge

# Create a DF to save the results (for plotting purposes)
aic_df = pd.DataFrame(aic_matrix, index=range(max_p + 1), columns=range(max_q + 1))
bic_df = pd.DataFrame(bic_matrix, index=range(max_p + 1), columns=range(max_q + 1))
hqic_df = pd.DataFrame(hqic_matrix, index=range(max_p + 1), columns=range(max_q + 1))

hqic_df[0][0] = np.nan
bic_df[0][0] = np.nan
# Find the lowest for each metric and extract its coordinates
min_aic = aic_df.stack().min()
min_aic_idx = np.unravel_index(np.nanargmin(aic_matrix), aic_matrix.shape)
min_bic = bic_df.stack().min()
min_bic_idx = np.unravel_index(np.nanargmin(bic_matrix), bic_matrix.shape)
min_hqic = hqic_df.stack().min()
min_hqic_idx = np.unravel_index(np.nanargmin(hqic_matrix), hqic_matrix.shape)

# Plot the heatmaps with Seaborn
# AIC
plt.figure(figsize=(10, 8))
sns.heatmap(aic_df, annot=True, cmap='viridis', fmt='.1f', cbar=True)
plt.title('Heatmap AIC for ARMA(p,q) models', size=18, pad=15)
plt.xlabel('q', size=16, labelpad=15)
plt.ylabel('p', size=16, labelpad=15)
rect = plt.Rectangle((min_aic_idx[1], min_aic_idx[0]), 1, 1, fill=False, edgecolor='orangered', linewidth=2)
plt.gca().add_patch(rect)
plt.show()

# BIC 
plt.figure(figsize=(10, 8))
sns.heatmap(bic_df, annot=True, cmap='viridis', fmt='.1f', cbar=True)
plt.title('Heatmap BIC for ARMA(p,q) models', size=18, pad=15)
plt.xlabel('q', size=16, labelpad=15)
plt.ylabel('p', size=16, labelpad=15)
rect = plt.Rectangle((min_bic_idx[1], min_bic_idx[0]), 1, 1, fill=False, edgecolor='orangered', linewidth=2)
plt.gca().add_patch(rect)
plt.show()

# HQIC 
plt.figure(figsize=(10, 8))
sns.heatmap(hqic_df, annot=True, cmap='viridis', fmt='.1f', cbar=True)
plt.title('Heatmap HQIC for ARMA(p,q) models', size=18, pad=15)
plt.xlabel('q', size=16, labelpad=15)
plt.ylabel('p', size=16, labelpad=15)
rect = plt.Rectangle((min_hqic_idx[1], min_hqic_idx[0]), 1, 1, fill=False, edgecolor='orangered', linewidth=2)
plt.gca().add_patch(rect)
plt.show()


# ARMA(3,7)
model_ar_3_ma_7 = ARIMA(close_train_diff, order=(3,0,7))
model_ar_3_ma_7_fit = model_ar_3_ma_7.fit()
model_ar_3_ma_7_fit.summary()

residui_ar_3_ma_7 = model_ar_3_ma_7_fit.resid
var = residui_ar_3_ma_7.var()
sd = np.sqrt(var)
mean = residui_ar_3_ma_7.mean()

plt.rcParams.update({'figure.figsize':(15,7)})
fig, ax = plt.subplots(1,2)
residui_ar_3_ma_7.plot(ax=ax[0], c='blue'); ax[0].set_title('Time plot', size=20); ax[0].set_xlabel('Date', size=18); ax[0].set_ylabel('Return %', size=18)
residui_ar_3_ma_7.plot(kind='hist', bins=50, ax=ax[1], color='blue'); ax[1].set_title('Histogram', size=20); ax[1].set_xlabel('Return %', size=18); ax[1].set_ylabel('Frequency', size=18)
ax[0].axhline(y=mean, color='red', linestyle='--', label=r'$\mu$', lw=2)
ax[1].axvline(x=sd, color='red', linestyle='--', label=r'+1 $\sigma$', lw=2)
ax[1].axvline(x=-sd, color='red', linestyle='--', label=r'-1 $\sigma$', lw=2)
ax[1].axvline(x=mean, color='orange', linestyle='--', label=r'$\mu$', lw=2)
ax[0].legend()
ax[1].legend()
plt.show()

plt.rcParams.update({'figure.figsize':(10,7)})
plot_acf(residui_ar_3_ma_7,auto_ylims=True, lags=40, zero=False, c='blue',title=None)
plt.xlabel('Lag', size=16)
plt.show()

# ARMA(2,2)
model_ar_2_ma_2 = ARIMA(close_train_diff, order=(2,0,2))
model_ar_2_ma_2_fit = model_ar_2_ma_2.fit()
model_ar_2_ma_2_fit.summary()

residui_ar_2_ma_2 = model_ar_2_ma_2_fit.resid
var = residui_ar_2_ma_2.var()
sd = np.sqrt(var)
mean = residui_ar_2_ma_2.mean()

plt.rcParams.update({'figure.figsize':(15,7)})
fig, ax = plt.subplots(1,2)
residui_ar_2_ma_2.plot(ax=ax[0], c='blue'); ax[0].set_title('Time plot', size=18); ax[0].set_xlabel('Date', size=16); ax[0].set_ylabel('Return %', size=16)
residui_ar_2_ma_2.plot(kind='hist', bins=50, ax=ax[1], color='blue'); ax[1].set_title('Histogramma', size=18); ax[1].set_xlabel('Return %', size=16); ax[1].set_ylabel('Frequency', size=16)
ax[0].axhline(y=mean, color='red', linestyle='--', label=r'$\mu$', lw=2)
ax[1].axvline(x=sd, color='red', linestyle='--', label=r'+1 $\sigma$', lw=2)
ax[1].axvline(x=-sd, color='red', linestyle='--', label=r'-1 $\sigma$', lw=2)
ax[1].axvline(x=mean, color='orange', linestyle='--', label=r'$\mu$', lw=2)
ax[0].legend()
ax[1].legend()
plt.show()

plt.rcParams.update({'figure.figsize':(10,7)})
plot_acf(residui_ar_2_ma_2,auto_ylims=True, lags=40, zero=False, c='blue',title=None)
plt.xlabel('Lag', size=18)
plt.show()

# ARMA(2,5)
model_ar_2_ma_5 = ARIMA(close_train_diff, order=(2,0,5))
model_ar_2_ma_5_fit = model_ar_2_ma_5.fit()
model_ar_2_ma_5_fit.summary()

residui_ar_2_ma_5 = model_ar_2_ma_5_fit.resid
var = residui_ar_2_ma_5.var()
sd = np.sqrt(var)
mean = residui_ar_2_ma_5.mean()

plt.rcParams.update({'figure.figsize':(15,7)})
fig, ax = plt.subplots(1,2)
residui_ar_2_ma_5.plot(ax=ax[0], c='blue'); ax[0].set_title('Time plot', size=18); ax[0].set_xlabel('Date', size=16); ax[0].set_ylabel('Return %', size=16)
residui_ar_2_ma_5.plot(kind='hist', bins=50, ax=ax[1], color='blue'); ax[1].set_title('Histogramma', size=18); ax[1].set_xlabel('Return %', size=16); ax[1].set_ylabel('Frequency', size=16)
ax[0].axhline(y=mean, color='red', linestyle='--', label=r'$\mu$', lw=2)
ax[1].axvline(x=sd, color='red', linestyle='--', label=r'+1 $\sigma$', lw=2)
ax[1].axvline(x=-sd, color='red', linestyle='--', label=r'-1 $\sigma$', lw=2)
ax[1].axvline(x=mean, color='orange', linestyle='--', label=r'$\mu$', lw=2)
ax[0].legend()
ax[1].legend()
plt.show()

plt.rcParams.update({'figure.figsize':(10,7)})
plot_acf(residui_ar_2_ma_5,auto_ylims=True, lags=40, zero=False, c='blue')
plt.xlabel('Lag', size=16)
plt.show()


# Calculate the forecasts for each model:
forecast_ar_1 = model_ar_1_fit.forecast(len(close_test))
forecast_ar_9 = model_ar_9_fit.forecast(len(close_test))
forecast_ma_1 = model_ma_1_fit.forecast(len(close_test))
forecast_ma_10 = model_ma_10_fit.forecast(len(close_test))
forecast_ma_14 = model_ma_14_fit.forecast(len(close_test))
forecast_ar_2_ma_2 = model_ar_2_ma_2_fit.forecast(len(close_test))
forecast_ar_2_ma_5 = model_ar_2_ma_5_fit.forecast(len(close_test))
forecast_ar_3_ma_7 = model_ar_3_ma_7_fit.forecast(len(close_test))

# Extract the last available price of the train set 
last_price = close_train['Close'].iloc[-1]

# Trasform the forecasts (Daily percentage returns) in prices:
price_forecast_ar_1 = last_price * (1 + forecast_ar_1 / 100).cumprod()
price_forecast_ar_9 = last_price * (1 + forecast_ar_9 / 100).cumprod()
price_forecast_ma_1 = last_price * (1 + forecast_ma_1 / 100).cumprod()
price_forecast_ma_10 = last_price * (1 + forecast_ma_10 / 100).cumprod()
price_forecast_ma_14 = last_price * (1 + forecast_ma_14 / 100).cumprod()
price_forecast_ar_2_ma_2 = last_price * (1 + forecast_ar_2_ma_2 / 100).cumprod()
price_forecast_ar_2_ma_5 = last_price * (1 + forecast_ar_2_ma_5 / 100).cumprod()
price_forecast_ar_3_ma_7 = last_price * (1 + forecast_ar_3_ma_7 / 100).cumprod()

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
plt.legend(fontsize=12)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

# Calculate the mean square error of the forecasts
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