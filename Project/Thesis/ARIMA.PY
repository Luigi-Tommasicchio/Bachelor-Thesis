import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# Importo i dati e seleziono il range di date desiderato
df = pd.read_csv('Thesis\\aapl_with_sentiment.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

print(df.head())

# Calcolo i retunrs e rimuovo la prima riga
df['Return'] = df['Close'].pct_change() * 100  
df.dropna(inplace=True)
print(df.head())

# Effettuo la decomposizione stagionale
result = seasonal_decompose(df['Close'], model='additive', period=4)
df['Seasonal'] = result.seasonal
df.head(100)
plt.show()

# Plot ACF e PACF per identificare le autocorrelazioni
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(df['Return'], lags=40, ax=ax1, zero=False)
plot_pacf(df['Return'], lags=40, ax=ax2, zero=False)
plt.show()

