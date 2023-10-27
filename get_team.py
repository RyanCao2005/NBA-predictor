import requests
import time
import requests
import pandas as pd
from time import sleep
import os
import json

def get_team():

  def api_request_with_retry(url, max_retries=3, initial_wait=1, max_wait=16):
      retries = 0

      while retries < max_retries:
          try:
              response = requests.get(url)
              response.raise_for_status()  # Raise an exception for HTTP errors
              return response  # Successful response, no need to retry

          except requests.exceptions.RequestException as e:
              print(f"Request failed: {e}")
              retries += 1
              if retries < max_retries:
                  wait_time = min(initial_wait * 2 ** retries, max_wait)
                  print(f"Retrying in {wait_time} seconds...")
                  time.sleep(wait_time)
              else:
                  print(f"Max retries ({max_retries}) reached. Giving up.")
                  return None
  ## handles errors









  ## web scraper for
  endpoint = 'https://free-nba.p.rapidapi.com/'
  headers = {
    "X-RapidAPI-Key": "d462d97e1dmsh591bce9df40272bp1029ddjsn2439341caddc",
    "X-RapidAPI-Host": "free-nba.p.rapidapi.com"
  }

  players_endpoint = os.path.join(endpoint, 'players')

  querystring = {'page': 0, 'per_page': 50}

  responses = []
  max_pages = float('inf')
  i = 0
  while i < max_pages:
    querystring['page'] = i
    response = requests.get(players_endpoint, headers=headers, params=querystring)
    while response.status_code == 429:
      print(f'Uh oh! We got put in time out >.<')
      sleep(10)
      response = requests.get(players_endpoint, headers=headers, params=querystring)
    response = json.loads(response.text)

    if max_pages == float('inf'):
      max_pages = response['meta']['total_pages']
    i += 1
    responses.extend(response['data'])
    print(f'Succesfully collected page {i}.')
    df = pd.DataFrame(responses)
    

  def extract_team_from_dict(d):
      if 'abbreviation' in d:
          return str(d['abbreviation'])
      else:
          return ''
  def extract_team_id_from_dict(d):
      if 'id' in d:
          return str(d['id'])
      else:
          return ''  
  df['team_id'] = df['team'].apply(extract_team_id_from_dict)      
  df['team'] = df['team'].apply(extract_team_from_dict)

  return df
get_team()
