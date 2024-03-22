import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from urllib.request import urlopen, Request
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Carica il DataFrame dal file CSV
df = pd.read_csv('Project/aapl_headlines.csv')
df['Data'] = pd.to_datetime(df['Data'])

# Inizializza l'analizzatore di sentimenti VADER
vader = SentimentIntensityAnalyzer()

# Definisci una funzione per calcolare il compound sentiment con VADER
f = lambda title: vader.polarity_scores(title)['compound']

# Calcola il compound sentiment per ogni titolo
df['compound'] = df['Titolo'].apply(f)

# Trasforma la colonna 'Data' in tipo data
df['Data'] = pd.to_datetime(df['Data']).dt.date

# Raggruppa per data e calcola la media del compound sentiment
mean_df = df.groupby('Data')['compound'].mean()

print(mean_df)

# Plot
plt.figure(figsize=(20, 10))
colors = mean_df.apply(lambda x: 'green' if x >= 0 else 'red')  
mean_df.plot(kind='bar', color=colors)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(100))
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(15))
plt.grid(True, linestyle=':', linewidth=0.5) 

# Crea gli oggetti Patch per la legenda
green_patch = mpatches.Patch(color='green', label='Positive Sentiment')
red_patch = mpatches.Patch(color='red', label='Negative Sentiment')

# Aggiungi i Patch alla legenda
plt.legend(handles=[green_patch, red_patch])
plt.show()

mean_df.to_csv('aapl_sentiment.csv')

