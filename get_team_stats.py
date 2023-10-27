import pandas as pd
import requests
from time import sleep
import json

#change this to save to your file path
save_path = ""
url = "https://free-nba.p.rapidapi.com/stats"

querystring = {"page":"0","per_page":"25"}

headers = {
	"X-RapidAPI-Key": "514d4e4fbcmsh9479ac730abbeb6p1d8bc3jsn7ef348d418d9",
	"X-RapidAPI-Host": "free-nba.p.rapidapi.com"
}

responses = []
max_pages = float('inf')
i = 0
while i < max_pages:
  querystring['page'] = i
  response = requests.get(url, headers=headers, params=querystring)
  while response.status_code == 429:
    print(f'Uh oh! We got put in time out >.<')
    sleep(10)
    response = requests.get(url, headers=headers, params=querystring)
  response = json.loads(response.text)
  if max_pages == float('inf'):
    max_pages = response['meta']['total_pages']
  i += 1
  responses.extend(response['data'])
  print(f'Succesfully collected page {i}.')

  df = pd.json_normalize(responses)
  df.to_csv(save_path, index=False)

