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



assets = ['MSFT','AAPL','TSLA','GOOG']

pf_data = pd.DataFrame()

pf_data = yf.download(sp500_tickers, start='2024-01-01', end='2024-04-01', interval='1d')['Adj Close']

del pf_data['BRK.B']
del pf_data['SOLV']
del pf_data['GEV']
del pf_data['BF.B']

pf_data.isna().sum().sum()

(pf_data / pf_data.iloc[0]*100).plot(figsize=(15,6))
plt.show()

log_returns = np.log(pf_data / pf_data.shift(1))

log_returns.mean()*250

log_returns.cov()*250

log_returns.corr()

num_assets = len(sp500_tickers)-4
num_assets

weights = np.random.random(num_assets)
weights /= np.sum(weights)
weights

np.sum(weights * log_returns.mean())*252

np.dot(weights.T, np.dot(log_returns.cov() *252, weights))

print(f"Expected Portfolio Return: {round(np.sum(weights * log_returns.mean())*252*100,2)}%")
print(f"Expected Portfolio Variance: {round(100*np.dot(weights.T, np.dot(log_returns.cov() *252, weights)),2)}%")
print(f"Expected Portfolio Volatility: {round(100*np.sqrt(np.dot(weights.T, np.dot(log_returns.cov()*252, weights))),2)}%")

np.sqrt(np.dot(weights.T, np.dot(log_returns.cov()*252, weights)))

pfolio_returns = []
pfolio_volatilities = []

for x in range(1000):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    
    pfolio_returns.append(np.sum(weights*log_returns.mean())*252)
    pfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(log_returns.cov()*250, weights))))
    
pfolio_returns = np.array(pfolio_returns)
pfolio_volatilities = np.array(pfolio_volatilities)

portfolios = pd.DataFrame({'Return':pfolio_returns,'Volatility':pfolio_volatilities})
portfolios.head()

portfolios.plot(x='Volatility',y='Return', kind='scatter', figsize=(10,6), alpha=0.5)
#plt.axis([0,])
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')
plt.show()













pf_data = yf.download('^GSPC', start='2024-01-01', end='2024-04-01', interval='1d')['Adj Close']

returns  = (pf_data.pct_change(1)*100)[1:]
returns

expected_return = sum(returns*(1/len(pf_data)))
expected_return

pf_data.isna().sum().sum()

(pf_data / pf_data.iloc[0]*100).plot(figsize=(15,6))
plt.show()

log_returns = np.log(pf_data / pf_data.shift(1))

log_returns.mean()*250

log_returns.cov()*250

log_returns.corr()

num_assets = 1
num_assets

weights = np.random.random(num_assets)
weights /= np.sum(weights)
weights

np.sum(weights * log_returns.mean())*252

np.dot(weights.T, np.dot(log_returns.cov() *252, weights))

print(f"Expected Portfolio Return: {round(np.sum(weights * log_returns.mean())*252*100,2)}%")
print(f"Expected Portfolio Variance: {round(100*np.dot(weights.T, np.dot(log_returns.cov() *252, weights)),2)}%")
print(f"Expected Portfolio Volatility: {round(100*np.sqrt(np.dot(weights.T, np.dot(log_returns.cov()*252, weights))),2)}%")

np.sqrt(np.dot(weights.T, np.dot(log_returns.cov()*252, weights)))

pfolio_returns = []
pfolio_volatilities = []

for x in range(1000):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    
    pfolio_returns.append(np.sum(weights*log_returns.mean())*252)
    pfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(log_returns.cov()*250, weights))))
    
pfolio_returns = np.array(pfolio_returns)
pfolio_volatilities = np.array(pfolio_volatilities)

portfolios = pd.DataFrame({'Return':pfolio_returns,'Volatility':pfolio_volatilities})
portfolios.head()

portfolios.plot(x='Volatility',y='Return', kind='scatter', figsize=(10,6), alpha=0.5)
#plt.axis([0,])
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')
plt.show()
