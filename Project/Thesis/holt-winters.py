import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

apple = pd.read_csv('Project\Thesis\AAPL.csv')
apple.head()

# Creazione e addestramento del modello Holt-Winters
split = int(len(apple)*0.8)

apple_train = apple[:split]
apple_test = apple[split:]

len(apple_test)
apple_train.tail()
apple_test.head()

model = ExponentialSmoothing(apple_train['Close'], trend='add', seasonal='mul', seasonal_periods=252)
hw_model = model.fit()
hw_model.summary()


# Fai previsioni per il periodo desiderato
forecast_period = len(apple_test)  # Esempio: prevedi i prossimi 30 giorni
forecast = hw_model.forecast(steps=forecast_period)
date = apple_test.index
forecasts = pd.DataFrame({'Date': date,'forecast': forecast})
forecasts.set_index('Date', inplace = True)
print(forecasts.head())
print(apple_test['Close'].head())

# Plot dei dati storici e delle previsioni
plt.figure(figsize=(12, 6))
plt.plot(apple_train['Close'], label='Train Closeing Price (Actual)')
plt.plot(apple_test['Close'], label='Test Closing Price (Actual)')
plt.plot(forecasts['forecast'], label='Close Price (Forecast)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Apple Close Price Forecast using Holt-Winters')
plt.legend()
plt.show()