import pandas as pd
import requests
from time import sleep
import json


save_path = "out.csv" #save path
last_page_file = "last.txt"  # File to save the last successful page number

url = "https://free-nba.p.rapidapi.com/stats"

querystring = {"page": "0", "per_page": "25"}

headers = {
    "X-RapidAPI-Key": "514d4e4fbcmsh9479ac730abbeb6p1d8bc3jsn7ef348d418d9",
    "X-RapidAPI-Host": "free-nba.p.rapidapi.com"
}

retries = 0
responses = []
max_pages = float('inf')
i = 0

while i < max_pages:
    querystring['page'] = i
    try:
        response = requests.get(url, headers=headers, params=querystring)
        while response.status_code == 429:
            print(f'Error 429 Retrying in 10 | Retries: {retries}')
            retries += 1
            sleep(10)
            response = requests.get(url, headers=headers, params=querystring)
        response = json.loads(response.text)
        if max_pages == float('inf'):
            max_pages = response['meta']['total_pages']
        responses.extend(response['data'])
        print(f'Successfully collected page {i}.')
        df = pd.json_normalize(responses)
        df.to_csv(save_path, index=False)
    except Exception as e:
        print(f"An error occurred at page {i}. Error: {e}")
        with open(last_page_file, "w") as file:
            file.write(str(i))
        break

    i += 1
