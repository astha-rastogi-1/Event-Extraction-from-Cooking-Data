### IMPORTS
import requests
from bs4 import BeautifulSoup
import time
from random import randint
import pandas as pd
import concurrent.futures

def get_url_data(url):
        ## Setting up the driver
        resp = requests.get(url)
        time.sleep(randint(5,15))
        soup = BeautifulSoup(resp.content, 'html.parser')

        ## Finding the name, ingredients and method from the page
        name = soup.find('h1', class_='headline heading-content')
        ingredients = soup.find('ul', class_='ingredients-section')
        procedure = soup.find('ul', class_='instructions-section')

        ## Add the data from that page to a list called data of size 3
        data = []
        data.append(' '.join(name.text.split()))
        data.append(' '.join(procedure.text.split()))
        data.append(' '.join(ingredients.text.split()))
        ## Creating a temp table with the values of data
        df = pd.DataFrame(columns=('Name','Method', 'Ingredients', 'Category', 'URL'))
        df['Name'] = [data[0]]
        df['Method'] = [data[1]]
        df['Ingredients'] = [data[2]]
        df['Category'] = ['fry, deep-fry']  # Replace categories
        df['URL'] = [url]
        ## Append to file
        df.to_csv('deep_fry.csv', mode='a', header=False)   # Replace with your file name
        del df


BASE_URL = "URL link here"
STORY_LINKS = []
MAX_THREADS = 30

if __name__=="__main__":
    # Get URLS
    for i in range(1,142):
        t0 = time.time()
        resp = requests.get(f"{BASE_URL}{i}")
        soup = BeautifulSoup(resp.content, "html.parser")
        time.sleep(randint(5,20))
        urls = [item.get("href") for item in soup.find_all("a")]
        ### To remove duplicate urls
        urls_final = list(dict.fromkeys(urls))
        ### To remove None values
        urls_final = list(filter(None, urls_final))

        ### To sort through the list of URLs to get only those that we want
        url_final = [x for x in urls_final if x.startswith('wanted url prefix')]    # Replace with the wanted URL prefix

        url_final = url_final[1:]


        t1 = time.time()
        print(f"{t1-t0} seconds for urls on page{i}")

        threads = min(MAX_THREADS, len(url_final))

        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            executor.map(get_url_data, url_final)
        t2 = time.time()
        print(f"{t2-t1} time for getting data of {len(url_final)} recipes on page {i}")