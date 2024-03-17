# Importo le funzioni necessarie
from scraper import scrape_news_headlines, df_from_scrape

# Dichiaro il ticker
ticker = 'aapl'

# Inserisco il ticker nel link
url = f'https://www.nasdaq.com/market-activity/stocks/{ticker}/news-headlines'

# Scarico le headlines degli articoli dal sito nasdaq.com e le assegno ad una lista
headlines = scrape_news_headlines(url, max_iterations=1250)

print(type(headlines))
print(headlines[:1])        # Stampo il primo elemento della lista per capire il tipo
                            # stringa che dobbiamo formattare

# Formatto le headlines separando la data dal titolo della notizia e creo un dataframe
# con una colonna per la data ed una colonna per il titolo
aapl_headlines = df_from_scrape(headlines)

# Stampo le prime righe del dataframe
print(aapl_headlines.head())

# Salvo il dataframe come csv
aapl_headlines.to_csv('aapl_headlines.csv',index=False)