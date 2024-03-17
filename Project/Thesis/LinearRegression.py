import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Importo i dati
df = pd.read_csv('AAPL.csv')

# Calcolo i Returns in percentuale
df['Return'] = df['Close'].pct_change() * 100  

# Creo la colonna Returns_t_minus_1 traslando in avanti di 1 i valori, lo stesso per i prezzi
df['Return_t_minus_1'] = df['Return'].shift(1)
print(df.head())

# Rimuovo le righe del dataset con valori NaN
df.dropna(inplace=True)
print(df.head())

# Divido il dataset in training e test sets
split_index = int(len(df) * 0.8)  
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

# Definisco le variabili indipendenti e dipendenti per training e test sets
X_train = train_df[['Return_t_minus_1']]
y_train = train_df['Return']

X_test = test_df[['Return_t_minus_1']]
y_test = test_df['Return']

# Aggiungo un termine costante alle variabili indipendenti per l'intercetta
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# Fit del modello di regressione lineare sul training set
model = sm.OLS(y_train, X_train).fit()

print(model.summary())

# Eseguo le previsioni sul test set
predicted_close_test = model.predict(X_test)

# Estraggo i coefficienti del modello
coefficients = model.params

# Creo l'equazione del modello
equation = f'y = {coefficients["const"]:.2f} {coefficients["Return_t_minus_1"]:.2f} * Return_t_minus_1'

# Plot temporale dei returns del training set, test set e previsioni
plt.figure(figsize=(12, 10))
plt.plot(train_df['Date'], train_df['Return'], color='blue', label='Train Set - Actual')
plt.plot(test_df['Date'], test_df['Return'], color='green', label='Test Set - Actual')
plt.plot(test_df['Date'], predicted_close_test, color='red', label='Test Set - Predicted')
plt.title('Actual vs. Predicted Returns')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.tick_params(axis='x', rotation=45)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(20))

# Aggiungi l'equazione del modello al grafico
plt.text(x='2022-09-13', y=-13, s=equation, fontsize=15)
plt.legend()
plt.show()