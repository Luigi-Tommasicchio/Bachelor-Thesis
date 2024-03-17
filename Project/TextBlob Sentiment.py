import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Importa il modulo nltk e scarica il dizionario per l'analisi dei sentimenti
import nltk
nltk.download('vader_lexicon')

# Carica i dati nel DataFrame di pandas (assumendo che il DataFrame sia chiamato df)
# Assicurati che il DataFrame contenga due colonne: 'data' per la data e 'titolo' per il titolo del giornale
# Esempio:
df = pd.read_csv('headlines.csv')

# Funzione per l'analisi dei sentimenti utilizzando TextBlob
def analyze_sentiment(title):
    analysis = TextBlob(title)
    # Ritorna il sentiment polarità (valore compreso tra -1 e 1)
    return analysis.sentiment.polarity

# Aggiungi una nuova colonna al DataFrame per il sentiment polarità
df['sentiment'] = df['Titolo'].apply(analyze_sentiment)
mean_df = df.groupby('Data')['sentiment'].mean()

# Stampare il DataFrame per visualizzare i risultati
print(mean_df)

# Plot
plt.figure(figsize=(20, 8))
mean_df.plot(kind='bar')
plt.grid(True)
plt.show()
