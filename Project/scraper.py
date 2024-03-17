from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def scrape_news_headlines(url, max_iterations=5):
    # Opzioni per il browser
    options = Options()
    options.headless = True  # Esegui il browser in modalità headless

    # Path del driver del browser
    chrome_driver_path = "C:\\Users\\luigi\\Desktop\\driver\\chromedriver.exe"  # Inserisci il percorso del tuo driver

    # Inizializza il browser
    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=options)

    # Lista per salvare tutte le news headlines
    all_headlines = []

    # Fai richiesta per ottenere la pagina
    driver.get(url)

    iteration = 0
    while iteration < max_iterations:
        # Aspetta 10 secondi per il caricamento delle headlines
        listacaricata = False
        while not listacaricata:
            try:
                time.sleep(0.02)

                # Trova tutti gli elementi li
                headlines_elements = driver.find_elements(By.CSS_SELECTOR, 'ul.quote-news-headlines__list li')

                # Estrai i testi delle headlines e aggiungili alla lista
                headlines_text = [element.text.strip() for element in headlines_elements if element.text.strip()]
                all_headlines.extend(headlines_text)

                listacaricata = True
            except Exception as e:
                print('Lista non caricata, ritento...')

                # Cerca il pulsante per passare alla pagina successiva
        bottonecliccato = False
        numero_tentativi = 0
        max_tentativi = 50

        while not bottonecliccato and numero_tentativi <= max_tentativi:
            try:
                next_page_button = WebDriverWait(driver, 0.01).until(
                     EC.element_to_be_clickable((By.CSS_SELECTOR, 'button.pagination__next')))

                # Verifica se il pulsante per la pagina successiva è abilitato
                if next_page_button.is_enabled():

                    driver.execute_script("arguments[0].scrollIntoView(true);", next_page_button)

                    # Fai clic sul pulsante per passare alla pagina successiva
                    next_page_button.click()
                    bottonecliccato = True
            except Exception as e:
                numero_tentativi +=1
                print(f"Bottone non cliccato, ritento...")
        if numero_tentativi == max_tentativi and not bottonecliccato:
            break

        iteration += 1
        print(iteration)

    # Chiudi il browser
    driver.quit()

    return all_headlines

'''
# URL della pagina delle news headlines
url = "https://www.nasdaq.com/market-activity/stocks/aapl/news-headlines"

# Chiama la funzione di scraping
news_headlines = scrape_news_headlines(url, max_iterations=1250)

# Stampa le news headlines
if news_headlines:
    for headline in news_headlines:
        print(headline)
else:
    print("Impossibile ottenere le news headlines.")


# Stampa il numero di headlines scaricate
n_headlines = len(news_headlines)
print('Numero Headlines: '+ str(n_headlines))

# Inizializza liste vuote per le colonne del DataFrame
#data_scraped = []
'''

# Estrai la prima linea, la seconda linea e il tempo da ogni stringa e aggiungili alle liste delle colonne
def df_from_scrape(news_headlines):
    titolo = []
    data_formattata = []

    for stringa in news_headlines:
        linee = stringa.split('\n')
        data = linee[0]
        titoli = linee[1]
        titolo.append(titoli)

        tempo_str = data.split()[0]
        if tempo_str.isdigit():
            tempo_unit = data.split()[1]
            if tempo_unit in ['MIN', 'MINS']:
                tempo = pd.to_datetime(datetime.now() - pd.Timedelta(minutes=int(tempo_str))).date()
            elif tempo_unit in ['HOUR', 'HOURS']:
                tempo = pd.to_datetime(datetime.now() - pd.Timedelta(hours=int(tempo_str))).date()
            elif tempo_unit in ['DAY', 'DAYS']:
                tempo = pd.to_datetime(datetime.now() - pd.Timedelta(days=int(tempo_str))).date()
            else:
                tempo = None
        else:
            tempo = pd.to_datetime(data, format='%b %d, %Y').date()
        data_formattata.append(tempo)

    df = pd.DataFrame({'Data': data_formattata, 'Titolo': titolo})
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.sort_values(by='Data')
    return df