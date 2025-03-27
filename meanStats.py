import pandas as pd
import numpy as np

# Assuming 'stats.csv' contains columns like 'game_id', 'player', 'points', 'assists', etc.
# playerstats_df = pd.read_csv('/stats10726.csv')

def get_player_and_date(game_id, dataframe):
    df = dataframe

    game = df[df['game.id'] == game_id]
    #get the list of players
    players = game['player.id'].unique()
    #organize the dataframe to only have games before that date
    target_Date = pd.to_datetime(game['game.date'].iloc[0])
    df['game.date'] = pd.to_datetime(df['game.date'])
    dateDF = df[df['game.date'] < target_Date]

    return np.array([players, dateDF], dtype=object)


"""
    priorAverages:
        Get's all of the players from the game_id and all the games those players were in
        before that game, it combines all of their stats ['id' ... 'pts'] and averages it,
        then puts it into a 2D array of the games and the player averages (for some reason
        25).
        
    Args:
        game_id (integer): The game id that we want to pick.
        param2 (data frame): The data frame of the CSV (Should be normalized)

    Returns:
        return_type: Description of the return value.
            player_averages_array (data frame): The game id that we want to pick.
            hometeam_win (boolean): If the home team won for the specificed game_id.
    """
def priorAverages(game_id, playerstats_df):
    # Get the list of players and the date for the given game_id
    players, date_df = get_player_and_date(game_id,playerstats_df)

    # Extract statistical columns (excluding 'game_id' and 'player')
    stat_columns = ['id', 'ast', 'blk', 'dreb', 'fg3_pct', 'fg3a', 'fg3m', 'fg_pct', 'fga',
                    'fgm', 'ft_pct', 'fta', 'min','ftm', 'oreb', 'pf', 'pts', 'reb', 'stl',
                    'turnover', 'game.visitor_team_score','game.home_team_score', 'player.height_feet', 'player.height_inches', 'player.weight_pounds']

    # Assuming players are already sorted by team in the list
    # Initialize a list to store average stats for each player
    player_averages_list = []

    # Calculate the average of each statistical category for each player
    for player in players:
        # Filter data for the current player
        player_data = date_df[date_df['player.id'] == player]
        player_data['min'] = pd.to_numeric(player_data['min'], errors='coerce')

        # Check if there are any games for the player
        if not player_data.empty:
            # Initialize a 1D NumPy array for each player
            average_stats = np.zeros(len(stat_columns))

            # Calculate the average for each statistical category
            for i, stat in enumerate(stat_columns):
                average_stats[i] = player_data[stat].mean()

            # Append the result to the list
            player_averages_list.append(average_stats)

    # Convert the list of player averages into a 2D NumPy array
    player_averages_array = np.array(player_averages_list)
    hometeam_win = playerstats_df[playerstats_df['game.id'] == game_id]['game.home_team_score'].values[0] > playerstats_df[playerstats_df['game.id'] == game_id]['game.visitor_team_score'].values[0]
    return player_averages_array, hometeam_win

# Example usage:
# game_id_to_analyze = 34574
# result, game_date = priorAverages(game_id_to_analyze)
# print(result,game_date)
