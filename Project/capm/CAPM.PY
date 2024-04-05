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

    # Path to the ChromeDriver executable
    chrome_driver_path = 'chrome_drivers\\chromedriver_windows.exe'  # Change this to the actual path

    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run Chrome in headless mode, no GUI

    # Create a new instance of Chrome driver
    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # URL of the website
    url = "https://www.slickcharts.com/sp500"

    # Navigate to the page
    driver.get(url)

    # Wait for the table to load
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "table.table.table-hover")))

    # Find the table containing the tickers
    table = driver.find_element(By.CSS_SELECTOR, "table.table.table-hover")

    # Find all rows in the table (excluding the header row)
    rows = table.find_elements(By.TAG_NAME, "tr")[1:]

    # Loop through each row and extract the ticker symbol
    for row in rows:
        # Extract ticker symbol from the first column
        ticker = row.find_elements(By.TAG_NAME, "td")[2].text.strip()
        tickers.append(ticker)

    # Quit the browser
    driver.quit()

    return tickers

# Call the function to scrape the tickers
sp500_tickers = scrape_sp500_tickers()

# Print the list of tickers
print(sp500_tickers)


data = pd.DataFrame()

data = yf.download(sp500_tickers, start='2024-01-01', end='2024-04-01', interval='1d')['Adj Close']

del data['BRK.B']
del data['SOLV']
del data['GEV']
del data['BF.B']

data.isna().sum().sum()

(data / data.iloc[0]*100).plot(figsize=(15,6))
plt.show()

returns = data.pct_change(1)*100

returns.mean()*250

returns.cov()*250

returns.corr()

num_assets = len(sp500_tickers)-4
num_assets

weights = np.random.random(num_assets)
weights /= np.sum(weights)
weights = np.round(weights, 5)

np.sum(weights * returns.mean())*252

np.dot(weights.T, np.dot(returns.cov() *252, weights))

print(f"Expected Portfolio Return: {round(np.sum(weights * returns.mean())*252*100,2)}%")
print(f"Expected Portfolio Variance: {round(100*np.dot(weights.T, np.dot(returns.cov() *252, weights)),2)}%")
print(f"Expected Portfolio Volatility: {round(100*np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252, weights))),2)}%")

np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252, weights)))

pfolio_returns = []
pfolio_volatilities = []

for x in range(1000):
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
test = data[['KO','WMT']]

# Calcolo i rendimenti giornalieri per ciascuna azione
returns = test.pct_change(1)*100

# Calcolo il rendimento atteso e la covarianza dei rendimenti per tutte le azioni
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Numero di portafogli generati casualmente
num_portfolios = 10000

# Lista vuota per i risultati
portfolio_returns = []
portfolio_std_devs = []
weight = []

# Genero i portafogli casuali
for i in range(num_portfolios):
    # Genero pesi casuali per il portafoglio
    weights = np.random.random(len(mean_returns))
    weights /= np.sum(weights)
    weights = np.round(weights, 5)
    
    # Calcolo il rendimento e la deviazione standard del portafoglio
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    
    # Aggiungo i risultati alla lista
    weight.append(weights)
    portfolio_returns.append(portfolio_return)
    portfolio_std_devs.append(portfolio_std_dev)

# Converto le liste in un DataFrame pandas
portfolio_data = pd.DataFrame({'Return': portfolio_returns, 'Std Dev': portfolio_std_devs, 'Weights': weight})
portfolio_data

# pesi del portafoglio ottimale con minima standard deviation
portfolio_data.sort_values(by='Std Dev', ascending=True)[:1]['Weights']

# Plotta tutte le varie combinazioni di portafogli
min_std_dev = portfolio_data.sort_values(by='Std Dev', ascending=True)[:1]['Return']
min_std_dev = float(min_std_dev)
min_std_dev

# Portafogli inefficienti
inefficient_portfolio_data = portfolio_data[portfolio_data['Return'] < min_std_dev]
efficient_portfolio_data = portfolio_data[portfolio_data['Return'] >= min_std_dev]

plt.figure(figsize=(10, 6))
plt.scatter(efficient_portfolio_data['Std Dev'], efficient_portfolio_data['Return'], alpha=0.3, color='blue', label='Portafogli Efficienti')
plt.scatter(inefficient_portfolio_data['Std Dev'], inefficient_portfolio_data['Return'], alpha=0.3, color='red', label='Portafogli Inefficienti')
plt.axhline(y=min_std_dev, color='black', linestyle='--', lw=2)
plt.xlabel('Deviazione Standard')
plt.ylabel('Rendimento atteso')
plt.title('Tutte le varie combinazioni di portafogli')
plt.grid(True)
plt.legend()
plt.show()