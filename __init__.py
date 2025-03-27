import requests
import pandas as pd
from tqdm import tqdm_notebook
from time import sleep
import os
import json
import sklearn
from IPython.display import clear_output

#get games played by a certain team after a certain date
def get_team_games(df, date, team):
    df['date'] = pd.to_datetime(df['date'])
    filtered_df = df[(df['date'] >= date) & ((df['home_team_full_name'] == team) | (df['visitor_team_full_name'] == team))]
    return filtered_df
