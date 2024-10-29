from nba_api.stats.endpoints import LeagueGameLog
import pandas as pd

def fetch_nba_data(season):
    # Formato da temporada deve ser "2020-21" para a temporada 2020-2021
    game_log = LeagueGameLog(season=season)
    game_data = game_log.get_data_frames()[0]
    
    # Mostra os primeiros dados para inspeção
    print(game_data.head())
    
    # Salva o dataframe em um CSV (opcional)
    game_data.to_csv(f"nba_season_{season}_data.csv", index=False)
    return game_data

# Exemplo de uso: buscar dados da temporada 2020-21
season_data = fetch_nba_data("2020-21")
