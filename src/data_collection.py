# data_collection.py

from nba_api.stats.endpoints import LeagueGameLog, BoxScoreTraditionalV2
import pandas as pd
import mysql.connector
import time
from datetime import datetime
import re
from tqdm import tqdm 

def get_season_range(years_back):
    current_year = datetime.now().year
    start_year = current_year - years_back
    return [f"{year}-{str(year + 1)[-2:]}" for year in range(start_year, current_year)]

def fetch_season_data(season):
    game_log = LeagueGameLog(season=season)
    data = game_log.get_data_frames()[0]

    data['GAME_TYPE'] = 'Regular Season'  # Defina um valor padrão

    return data[['SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'MIN',
                 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 
                 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS', 'GAME_TYPE']]  # Inclua GAME_TYPE aqui

def fetch_player_data(game_id):
    try:
        box_score = BoxScoreTraditionalV2(game_id=game_id)
        player_stats = box_score.player_stats.get_data_frame()
        player_stats['GAME_ID'] = game_id
        return player_stats
    except Exception as e:
        print(f"Erro ao obter dados para o game_id {game_id}: {e}")
        return pd.DataFrame()

def clean_minutes_column(player_data):
    if 'MIN' in player_data.columns:
        player_data['MIN'] = player_data['MIN'].apply(lambda x: float(re.split("[:]", str(x))[0]) if pd.notnull(x) else 0.0)
    else:
        player_data['MIN'] = 0.0  # Se não houver a coluna MIN, define como 0
    return player_data

def insert_season_data_to_db(connection, season_data):
    cursor = connection.cursor()
    
    # Definindo VIDEO_AVAILABLE como 0 se não estiver presente
    if 'VIDEO_AVAILABLE' not in season_data.columns:
        season_data['VIDEO_AVAILABLE'] = 0

    season_data = season_data.drop(columns=['VIDEO_AVAILABLE'])  # Removendo coluna extra

    for _, row in season_data.iterrows():
        cursor.execute('''
            INSERT INTO season_data (SEASON_ID, TEAM_ID, TEAM_ABBREVIATION, TEAM_NAME, GAME_ID, GAME_DATE, MATCHUP, WL, MIN,
                                     FGM, FGA, FG_PCT, FG3M, FG3A, FG3_PCT, FTM, FTA, FT_PCT, OREB, DREB, REB, AST, STL,
                                     BLK, TOV, PF, PTS, PLUS_MINUS, GAME_TYPE)  -- Adicione GAME_TYPE aqui
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE WL=VALUES(WL), PTS=VALUES(PTS), PLUS_MINUS=VALUES(PLUS_MINUS)
        ''', tuple(row))

    connection.commit()

def insert_player_data_to_db(connection, player_data):
    cursor = connection.cursor()
    columns_to_drop = ['TEAM_ABBREVIATION', 'TEAM_CITY', 'NICKNAME', 'COMMENT']
    player_data = player_data.drop(columns=[col for col in columns_to_drop if col in player_data.columns])
    player_data = clean_minutes_column(player_data)  # Limpeza da coluna MIN
    for _, row in player_data.iterrows():
        cursor.execute('''
            INSERT INTO player_data (GAME_ID, TEAM_ID, PLAYER_ID, PLAYER_NAME, START_POSITION, MIN, FGM, FGA, FG_PCT,
                                     FG3M, FG3A, FG3_PCT, FTM, FTA, FT_PCT, OREB, DREB, REB, AST, STL, BLK, T_O, PF, PTS,
                                     PLUS_MINUS)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE PTS=VALUES(PTS), PLUS_MINUS=VALUES(PLUS_MINUS)
        ''', tuple(row))

    connection.commit()

def main():
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='SAFADO',
        database='nba_data'
    )

    all_time_seasons = get_season_range(75)  
    last_10_years_seasons = get_season_range(10)  
    last_5_years_seasons = get_season_range(5)  

    selected_seasons = last_5_years_seasons  

    for season in tqdm(selected_seasons, desc="Capturando dados de temporada"):
        print(f"\nProcessando temporada {season}...")
        season_data = fetch_season_data(season)
        insert_season_data_to_db(connection, season_data)

        game_ids = season_data['GAME_ID'].unique()
        for game_id in tqdm(game_ids, desc=f"Processando jogos da temporada {season}", leave=False):
            player_data = fetch_player_data(game_id)
            insert_player_data_to_db(connection, player_data)
            time.sleep(1)  # Pausa para evitar limite da API

    connection.close()

if __name__ == "__main__":
    main()
