import requests
import pandas as pd
from time import sleep
import os
import json

endpoint = 'https://free-nba.p.rapidapi.com/'
headers = {
	"X-RapidAPI-Key": "563b99e32emsh860d6965169a64dp1da03bjsnf21eaf721a45",
	"X-RapidAPI-Host": "free-nba.p.rapidapi.com"
}

players_endpoint = os.path.join(endpoint, 'games')

querystring = {'page': 0, 'per_page': 50}

responses = []
max_pages = float('inf')
i = 0
num_timeouts = 0
success = 0
while i < max_pages:
  querystring['page'] = i
  response = requests.get(players_endpoint, headers=headers, params=querystring)
  while response.status_code == 429:
    num_timeouts += 1
    print(f'Uh oh! We got put in time out >.< {num_timeouts}th time')
    sleep(10)
    response = requests.get(players_endpoint, headers=headers, params=querystring)
  response = json.loads(response.text)

  if max_pages == float('inf'):
    max_pages = response['meta']['total_pages']
  i += 1
  responses.extend(response['data'])
  success += 1

print('successfully collected data')
# at this point we should just have a csv file to put up 

df = pd.DataFrame(responses)

home_team_series = pd.DataFrame(df['home_team'].to_list())
home_team_series.columns = 'home_team_' + home_team_series.columns
df1 = pd.concat([df, home_team_series], axis=1)
df1 = df1.drop('home_team', axis=1)
columns = df1.columns.tolist()
new_columns = columns[:2] + columns[10:] + columns[2:10]
df2 = df1[new_columns]
visitor_team_series = pd.DataFrame(df['visitor_team'].to_list())
visitor_team_series.columns = 'visitor_team_' + visitor_team_series.columns
df3 = pd.concat([df2, visitor_team_series], axis = 1)
df3 = df3.drop('visitor_team', axis = 1)