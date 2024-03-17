# Importo i pacchetti necessari:
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.seasonal import seasonal_decompose


np.random.seed(1234)

# Importo i dati dell'S&P 500 utilizzando yfinance:
start = '2014-01-01'
end = '2024-01-01'
sp500 = yf.Ticker('^GSPC').history(start = start, end = end, interval='1d')['Close']
sp500 = sp500.reset_index()
sp500['Date'] = sp500['Date'].dt.date
sp500.set_index('Date', inplace=True)
print(sp500.tail())




# Creo un time plot:
sp500.Close.plot(figsize = (15,7), color='blue', label='Prezzo di Chiusura')
plt.title('Prezzi di Chiusura S&P 500 (2014-2024)', size=24, pad=15)
plt.yticks(size=15)
plt.xticks(size=15)
plt.xlim(sp500.index.min(), sp500.index.max())
plt.ylabel('Prezzo $', size=20, labelpad=13)
plt.xlabel('Data', size=20, labelpad=13)
plt.legend(fontsize=15)
plt.show()




# Creiamo una serie white noise e la plottiamo:
sp500['wn'] = np.random.normal(loc=sp500.Close.mean(), scale=sp500.Close.std(), size=len(sp500))

sp500.wn.plot(figsize = (15,7), color='blue', linewidth=.7, label = 'White Noise')
plt.title('Simulazione White Noise', size=24, pad=15)
plt.yticks(size=15)
plt.xticks(size=15)
plt.xlim(sp500.index.min(), sp500.index.max())
plt.ylabel('Prezzo $', size=20, labelpad=13)
plt.xlabel('Data', size=20, labelpad=13)
plt.legend(fontsize=15)
plt.show()

# Confrontiamo Close e WN:
print(sp500.describe())



# Generiamo una Random Walk:
def rw_gen(T = 1, N = 100, mu = 0.1, sigma = 0.01, S0 = 20):
    dt = float(T)/N
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size = N)
    W = np.cumsum(W)*np.sqrt(dt)
    X = (mu-0.5*sigma**2)*t + sigma*W
    S = S0*np.exp(X)
    return S

dates = pd.date_range('2014-01-01', '2024-01-01')
T = (dates.max()-dates.min()).days / 365
N = dates.size
start_price = sp500.Close[0]
rw = pd.Series(rw_gen(T, N, mu=0.07, sigma=0.1, S0=start_price), index=dates)

sp500['rw'] = rw

sp500.rw.plot(figsize = (15,7), color='blue', label = 'Random Walk')
plt.title('Simulazione Random Walk', size=24, pad=15)
plt.yticks(size=15)
plt.xticks(size=15)
plt.xlim(sp500.index.min(), sp500.index.max())
plt.ylabel('Prezzo $', size=20, labelpad=13)
plt.xlabel('Data', size=20, labelpad=13)
plt.legend(fontsize=15)
plt.show()



# Creazione della figura e dei tre assi:
fig, axs = plt.subplots(3, 1, figsize=(9, 13))

# Grafico 1:
axs[0].plot(sp500.index, sp500.Close, color='navy')
axs[0].set_title('Prezzi di Chiusura S&P 500 (2014-2024', size=21, pad=10)
axs[0].set_xlabel('Data', size=16)
axs[0].set_ylabel('Prezzo $', size=16)
axs[0].tick_params(axis='both', which='major', labelsize=14)

# Grafico 2:
axs[1].plot(sp500.index, sp500.wn, color='darkblue', linewidth=0.7)
axs[1].set_title('Simulazione White Noise', size=21, pad=10)
axs[1].set_xlabel('Data', size=16)
axs[1].set_ylabel('Prezzo $', size=16)
axs[1].tick_params(axis='both', which='major', labelsize=14)

# Grafico 3:
axs[2].plot(sp500.index, sp500.rw, color='mediumblue')
axs[2].set_title('Simulazione Random Walk', size=21, pad=10)
axs[2].set_xlabel('Data', size=16)
axs[2].set_ylabel('Prezzo $', size=16)
axs[2].tick_params(axis='both', which='major', labelsize=14)

# Imposto i limiti per l'asse x:
for ax in axs:
    ax.set_xlim(sp500.index.min(), sp500.index.max())
    ax.set_ylim(-500, 6000)

# Ottimizza il layout per evitare sovrapposizioni:
plt.tight_layout()

# Imposta lo spaziatura verticale tra i grafici:
plt.subplots_adjust(hspace=0.5)

# Mostra il grafico:
plt.show()




def adf_test(column):
    # Esecuzione del test di Dickey-Fuller Aumentato (ADF) per la colonna specificata
    adf_result = sts.adfuller(column)
    
    adf_output = pd.Series(adf_result[0:4], index=['Test Statistic','P-value','Lags Used','Number of Observations Used'])
    for key, value in adf_result[4].items():
        adf_output[f'Critical Value ({key})'] = value

    return adf_output

# Esecuzione del test di Dickey-Fuller Aumentato (ADF) per ciascuna serie
def adf_table(columns = []):
    table = pd.DataFrame()
    for i in range(0,len(columns)):
        adf = adf_test(sp500[str(columns[i])])
        table[str(columns[i])] = adf
    return table

adf_table = adf_table(columns=['Close', 'wn', 'rw'])
print(adf_table)

# Round dei valori nel DataFrame con un numero massimo di decimali
adf_table_rounded = adf_table.round(decimals=4)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

# Rimuovi gli assi
ax.axis('off')

# Crea la tabella con i valori arrotondati del DataFrame
table = ax.table(cellText=adf_table_rounded.values,
                 colLabels=adf_table_rounded.columns,
                 rowLabels=adf_table_rounded.index,
                 cellLoc='right',
                 loc='center',
                 bbox=[0.3, 0.3, 0.77, 0.5])  # Imposta la posizione e le dimensioni della tabella

# Imposta lo stile della tabella
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1)  # Imposta la dimensione della tabella

# Mostra la tabella
plt.show()