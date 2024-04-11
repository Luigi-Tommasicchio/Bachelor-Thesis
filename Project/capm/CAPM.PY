import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import yfinance as yf

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def scrape_sp500_tickers():
    tickers = []

    chrome_driver_path = 'chrome_drivers/chromedriver'  
    chrome_options = Options()
    chrome_options.add_argument('--headless')  

    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    url = "https://www.slickcharts.com/sp500"

    driver.get(url)

    # Aspetta finchè non appare la tabella con i vari ticker
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.table.table-hover")))

    # Trova la tabella corretta
    table = driver.find_element(By.CSS_SELECTOR, "table.table.table-hover")

    # Estrae tutte le riche della tabella ad eccezione dell'header
    rows = table.find_elements(By.TAG_NAME, "tr")[1:]

    # Fa un loop attraverso le righe per estrarre tutti i tickers
    for row in rows:
        # Estrae il ticker per ogni riga dalla terza colonna
        ticker = row.find_elements(By.TAG_NAME, "td")[2].text.strip()
        tickers.append(ticker)

    # Chiude chrome
    driver.quit()

    return tickers

# Chiamo la funzione per lo scraping dei tickers
sp500_tickers = scrape_sp500_tickers()

# Stampo la lista di tutti i tickers
print(sp500_tickers)

# Inizializzo un dataframe
data = pd.DataFrame()

# Scarico i prezzi ci chiusura per ciascun ticker
data = yf.download(sp500_tickers, start='2024-01-01', end='2024-04-01', interval='1d')['Adj Close']

# Elimino delle colonne con valori mancanti
del data['BRK.B']
del data['SOLV']
del data['GEV']
del data['BF.B']

# Controllo che non ci siano valori mancanti
data.isna().sum().sum()

(data / data.iloc[0]).plot(figsize=(15,6))
plt.show()

# Calcolo i rendimenti
returns = np.round((data.pct_change(1)*100),2)[1:]
returns['AAPL'].plot(kind='hist',bins=30, density=True, figsize=(15,7))
plt.show()

returns.mean()*252

returns.cov()*252

returns.corr()

num_assets = len(sp500_tickers)-4
num_assets

weights = np.random.random(num_assets)
weights /= np.sum(weights)
weights = np.round(weights, 5)

print(f"Expected Portfolio Return: {round(np.sum(weights * returns.mean())*252,2)}%")
print(f"Expected Portfolio Variance: {round(np.dot(weights.T, np.dot(returns.cov() *252, weights)),2)}%")
print(f"Expected Portfolio Volatility: {round(np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252, weights))),2)}%")



pfolio_returns = []
pfolio_volatilities = []

for x in range(10):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    
    pfolio_returns.append(np.sum(weights*returns.mean())*252)
    pfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(returns.cov()*250, weights))))
    
pfolio_returns = np.array(pfolio_returns)
pfolio_volatilities = np.array(pfolio_volatilities)

portfolios = pd.DataFrame({'Return':pfolio_returns,'Volatility':pfolio_volatilities})
portfolios.head()

portfolios.plot(x='Volatility',y='Return', kind='scatter', figsize=(10,6), alpha=0.5)
#plt.axis([0,])
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')
plt.show()






# Scegliamo le azioni da includere nel portafoglio
test = data[['AAPL', 'META']]

# Calcolo i rendimenti giornalieri per ciascuna azione
returns = test.pct_change(1)*100

# Calcolo il rendimento atteso e la covarianza dei rendimenti per tutte le azioni
mean_returns = returns.mean()
variance = returns.var()
standard_error = np.sqrt(variance)
cov_matrix = returns.cov()

# Numero di portafogli generati casualmente
num_portfolios = 10

# Lista vuota per i risultati
portfolio_returns = []
portfolio_std_dev = []
weight = []

# Genero i portafogli casuali
for i in range(num_portfolios):
    # Genero pesi casuali per il portafoglio
    weights = np.random.random(len(mean_returns))
    weights /= np.sum(weights)
    weights = np.round(weights, 5)
    
    # Calcolo il rendimento e la deviazione standard del portafoglio
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_std_d = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    
    # Aggiungo i risultati alla lista
    weight.append(weights)
    portfolio_returns.append(portfolio_return)
    portfolio_std_dev.append(portfolio_std_d)

# Converto le liste in un DataFrame pandas
portfolio_data = pd.DataFrame({'Return': portfolio_returns, 'Std Dev': portfolio_std_dev, 'Weights': weight})

# pesi del portafoglio ottimale con minima standard deviation
portfolio_data.sort_values(by='Std Dev', ascending=True)[:1]['Weights']

# Plotta tutte le varie combinazioni di portafogli
min_std_dev = portfolio_data.sort_values(by='Std Dev', ascending=True)[:1]['Return']
min_std_dev = float(min_std_dev)

# Portafogli inefficienti
inefficient_portfolio_data = portfolio_data[portfolio_data['Return'] < min_std_dev]
efficient_portfolio_data = portfolio_data[portfolio_data['Return'] > min_std_dev]
P_portfolio_data = portfolio_data[portfolio_data['Return'] == min_std_dev]


plt.figure(figsize=(10, 6))
plt.scatter(efficient_portfolio_data['Std Dev'], efficient_portfolio_data['Return'], alpha=1, color='blue', label='Portafogli Efficienti')
plt.scatter(inefficient_portfolio_data['Std Dev'], inefficient_portfolio_data['Return'], alpha=1, color='red', label='Portafogli Inefficienti')
plt.axhline(y=min_std_dev, color='black', linestyle='--', lw=1.5)
plt.scatter(P_portfolio_data['Std Dev'], P_portfolio_data['Return'], alpha=1, color='yellow', label='Portafoglo con minor volatilità', edgecolors='black')
plt.xlabel('Deviazione Standard')
plt.ylabel('Rendimento atteso')
plt.title('Tutte le varie combinazioni di portafogli')
plt.grid(True)
plt.xlim(0,portfolio_data['Std Dev'].max()+1)
plt.legend()
plt.show()