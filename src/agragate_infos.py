import pandas as pd

season_data = pd.read_csv('./data/nba_season_data_no_outliers.csv')
player_data = pd.read_csv('./data/nba_player_data_no_outliers.csv')

print("Colunas em 'season_data':", season_data.columns)
print("Colunas em 'player_data':", player_data.columns)

merged_data = pd.merge(player_data, season_data, on=["GAME_ID", "TEAM_ID"], how="left", suffixes=('_player', '_season'))

if 'TEAM_ID' not in merged_data.columns:
    raise KeyError("A coluna 'TEAM_ID' está ausente após a mesclagem.")

merged_data['TOTAL_PTS_TEAM'] = merged_data.groupby(['GAME_ID', 'TEAM_ID'])['PTS_player'].transform('sum')
merged_data['AVG_REB_TEAM'] = merged_data.groupby(['GAME_ID', 'TEAM_ID'])['REB_player'].transform('mean')
merged_data['AVG_AST_TEAM'] = merged_data.groupby(['GAME_ID', 'TEAM_ID'])['AST_player'].transform('mean')
merged_data['TOTAL_MIN_TEAM'] = merged_data.groupby(['GAME_ID', 'TEAM_ID'])['MIN_player'].transform('sum')

columns_to_keep = [
    'GAME_ID', 'TEAM_ID', 'PLAYER_ID', 'PLAYER_NAME', 'WL', 'PTS_player', 
    'REB_player', 'AST_player', 'MIN_player', 'TOTAL_PTS_TEAM', 
    'AVG_REB_TEAM', 'AVG_AST_TEAM', 'TOTAL_MIN_TEAM', 
    'FG_PCT_season', 'FG3_PCT_season', 'FT_PCT_season'
]

final_data = merged_data[columns_to_keep]

final_data.to_csv('./data/nba_detailed_merged_data.csv', index=False)
print("Dados mesclados e salvos em nba_detailed_merged_data.csv")
