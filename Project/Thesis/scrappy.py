from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd

def scrape_news_headlines(url, max_pages=5, scroll_amount=200):
    options = Options()
    options.headless = True

    chrome_driver_path = "C:/Users/luigi/Desktop/driver/chromedriver.exe"

    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=options)

    all_headlines = []

    for page in range(1, max_pages + 1):
        page_url = f"{url}?page={page}" if page > 1 else url
        driver.get(page_url)

        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "jupiter22-c-article-list__item_title")))

        # Esegui lo scroll verso il basso prima di passare alla pagina successiva
        driver.execute_script(f"window.scrollBy(0, {scroll_amount})")

        # Aspetta un po' dopo lo scroll per permettere il caricamento completo della pagina
        time.sleep(1)

        title_elements = driver.find_elements(By.CLASS_NAME, 'jupiter22-c-article-list__item_title')
        timeline_elements = driver.find_elements(By.CLASS_NAME, 'jupiter22-c-article-list__item_timeline')

        for title_element, timeline_element in zip(title_elements, timeline_elements):
            title = title_element.text
            timeline = timeline_element.text
            all_headlines.append({'Timeline': timeline, 'Title': title})

    driver.quit()

    return all_headlines

# URL della pagina da cui estrarre i titoli degli articoli
url = "https://www.nasdaq.com/market-activity/stocks/aapl/news-headlines"

# Numero massimo di pagine da scansionare
max_pages = 5

# Quantità di scroll in pixel
scroll_amount = 600

# Esegui lo scraping delle headline
headlines = scrape_news_headlines(url, 5, scroll_amount)

# Creazione del DataFrame
df = pd.DataFrame(headlines)

# Stampa del DataFrame
print(df)

from datetime import datetime, timedelta
import pandas as pd

def parse_timeline(timeline):
    if "ago" in timeline:
        parts = timeline.split()
        num = int(parts[0])
        unit = parts[1].lower()

        if "hour" in unit:
            delta = timedelta(hours=num)
        elif "day" in unit:
            delta = timedelta(days=num)
        else:
            delta = timedelta()

        return datetime.now() - delta
    else:
        try:
            return datetime.strptime(timeline, "%b %d, %Y")
        except ValueError:
            return None

# Eseguiamo il parsing degli elementi della colonna "Timeline"
df['Timeline'] = df['Timeline'].apply(parse_timeline)

# Stampa del DataFrame con la colonna "Timeline" riformattata
print(df)


from textblob import TextBlob

# Funzione per l'analisi del sentiment
def analyze_sentiment(title):
    blob = TextBlob(title)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score

# Applica l'analisi del sentiment a ogni titolo nel DataFrame
df['Sentiment'] = df['Title'].apply(analyze_sentiment)

# Stampa il DataFrame con la colonna del sentiment
print(df)

import matplotlib.pyplot as plt
df.Sentiment.plot()
plt.show()

import yfinance as yf

apple = yf.download('aapl')['Close']

returns = apple.pct_change(1)[1:]*100

returns.index[-2:]
returns[-20:].plot()
df.Sentiment.plot()
plt.show()

