import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as sts # for the .adfuller()

# Define the ticker, start and end date for the downlaod:
ticker = 'AAPL'
start = '2019-01-01'
end = '2024-01-01'

# Download the data setting the frequency to 1 day:
df = yf.download(ticker, start=start, end=end, interval='1d')

# Remove columns we don't need:
del df['Open']
del df['High']
del df['Low']
del df['Volume']
del df['Adj Close'] 

# Print the first 5 rows:
df.head()

# Create 2 columns, one with the differenced series and another just as a check for the integration:
df['close_diff'] = df.Close.diff(1)
df['close_int'] = np.nancumsum(df.close_diff) + df.Close[0]

# Let's visualize the series:
fig, ax = plt.subplots(2,1, sharex=True, figsize=(15,8))
ax[0].plot(df.Close, label='Original Series'); ax[0].plot(df.close_int, linestyle='--', label='Integrated Series'); ax[0].set_ylabel('Price $', size=16)
ax[1].plot(df.close_diff); ax[1].set_ylabel('1-day difference in price $', size=16)
ax[1].set_xlabel('Date', size=16)
fig.suptitle('Original and differenced series:', size=20, y=0.95)
ax[0].legend()
plt.show()

# Let's check whether the series are stationary or not:
sts.adfuller(df.Close) # <-- Not stationary
sts.adfuller(df.close_diff[1:]) # <-- Stationary

# Okay, so taking the 1st order difference worked, the series is now stationary and we know how to go back to the original one!

# Now the next step would be splitting the data into a train set and a test set.

# Afterward, as you can probably imagine, we'll:
# 1) Train the model on the train set;
# 2) Evaluate it; 
# 3) Produce the forecasts; 
# 4) Compare the forecasts with the actual observations from the test set.

# Let's start by extracting the 'close_diff' column into a pd.Series that we can name the same way: 
close_diff = df.close_diff
type(close_diff) # little check

# Now, let's say we wanna train the model on 80% of the data and test it on 20%, this is how we do it:
split = int(len(close_diff)*.8) # we create and index to split the series

train, test = close_diff[:split], close_diff[split:] # we create the 2 sets using the index
                                                           
train.tail() # we check if everything is alright
test.head()

# Let's visualize them:
train.plot(color='blue', label='Train set')
test.plot(color='red', label='Test set')
plt.title('Splitted series ready for modeling:')
plt.ylabel('1-day differenced prices, squared ($)')
plt.legend()
plt.show()

# Let's check again the stationarity just to be sure.
sts.adfuller(train[1:]) # The squared differences ad still stationary, wonderful!

# Now let's import 2 super useful function from the statsmodels library:
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Let's plot the ACF for the train set:
fig, ax = plt.subplots(1,1, figsize=(10,5))
plot_acf(train[1:], auto_ylims=True, lags=40, ax=ax)
plt.show()

# Let's now define the model:
from arch import arch_model

bic = []
for i in range(1,40): 
    model = arch_model(train[1:], mean='Zero', vol='ARCH', p=1)
    model_fit = model.fit()
    bic.append(model_fit.bic)
bic = pd.Series(bic)
bic.plot()
plt.show()

model = arch_model(train[1:])
model_fit = model.fit()
model_fit.summary()

yhat = model_fit.forecast(horizon=len(test))
yhat.variance.values[-1,:]
forecasts = pd.Series(yhat.mean.values[-1,:])
plt.plot(test.index, forecasts)
test.plot()
plt.show()

