# Importo le librerie necessarie
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Download dei dati da Yahoo Finance
ticker = 'AAPL'
start_date = '2014-01-01'
end_date = '2024-01-01'

apple = yf.Ticker(ticker).history(ticker, start=start_date, end=end_date, interval='1d')[['Open', 'Close', 'High', 'Low', 'Volume']]

apple.head()

# Controllo il datatype delle colonne
apple.info()
# Controllo se ci sono valori nulli
apple.isna().sum()

# Riformatto l'indice da 'datetime' a 'date'
apple = apple.reset_index()
apple['Date'] = pd.to_datetime(apple['Date']).dt.date
apple.set_index('Date', inplace=True)
apple.head()

