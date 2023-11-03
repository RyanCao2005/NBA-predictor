import requests
import pandas as pd
from tqdm import tqdm_notebook
from time import sleep
import os
import json
from IPython.display import clear_output

#this part gets the games information from the api
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
    if(num_timeouts % 20 == 0):   
        print(f'Uh oh! We got put in time out >.< {num_timeouts}th time')
    sleep(10)
    response = requests.get(players_endpoint, headers=headers, params=querystring)
  response = json.loads(response.text)

  if max_pages == float('inf'):
    max_pages = response['meta']['total_pages']
  i += 1
  responses.extend(response['data'])
  success += 1
  print(f'collected {success} pages')
  clear_output(wait = True)

print('successfully collected data')


#put the data into a dataframe, and do some data cleaning
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
cleaned_df = df3.drop('visitor_team', axis = 1)
cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])
#the csv file is also available, should be in the repo
cleaned_df.to_csv('games.csv')


#normalizing columns using min-max scaling
def normalize_column(df, column_name):
    # Check if the column exists in the DataFrame
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found in the DataFrame.")
        return df  # Return the original DataFrame

    min_val = df[column_name].min()
    max_val = df[column_name].max()

    if min_val == max_val:
        # Handle the case where min and max are the same to avoid division by zero
        df[column_name] = 0.0
    else:
        df[column_name] = (df[column_name] - min_val) / (max_val - min_val)

    return df


#only normalizing home_team_score and visitor_team_score as they are the only ones that require normalizing.
#it doesn't make sense to normalize the other int64 values, such as team_ids and season number,
#as those are there for identification purposes. 
normalized_df = normalize_column(cleaned_df, 'home_team_score')
normalized_df = normalize_column(normalized_df, 'visitor_team_score')

