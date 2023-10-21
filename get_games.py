from requests import get
import pandas as pd 
import numpy as np
import os
import json
import requests


#get player_ids dataframe 
response = get(r'https://bucketlist.fans/api/v1/player_names')
f = json.loads(response.text)
player_ids = pd.DataFrame(f).transpose()

def get_player_id(player_name, df):
    #takes name and player_id dataframe, returns playerid 
    names = player_name.split()
    first = names[0]
    last = names[1]
    id = df.index[(df['firstName'] == first) & (df['lastName'] == last)].tolist()
    return id 

def get_games(player_name=None, player_id = None):
    #takes playername or playerid and returns df of games 
    
    if player_name:
        player_id = int(get_player_id(player_name,player_ids)[0])
    if not player_id:
        raise ValueError
    
    url = r'https://bucketlist.fans/api/v1/player_game_log'
    payload = {'playerId' : player_id}
    headers = {
        'Origin': 'https://bucketlist.fans',
        'Referer': 'https://bucketlist.fans/',
        'Content-Type': 'application/json'
    }
    player_response = requests.post(url, headers=headers, json = payload)
    f = json.loads(player_response.text)
    return pd.DataFrame(f)