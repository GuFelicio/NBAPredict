import pandas as pd
import mysql.connector
from sklearn.preprocessing import StandardScaler


def fetch_data_from_db():
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='USA SUA',
        database='nba_data'
    )

    season_data = pd.read_sql("SELECT * FROM season_data", connection)
    player_data = pd.read_sql("SELECT * FROM player_data", connection)

    connection.close()
    return season_data, player_data

def handle_missing_values(df):
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        df[column] = df[column].fillna(0)  # Usando atribuição direta
    return df

def normalize_data(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def aggregate_player_data(player_data):
    # Limpeza dos dados dos jogadores
    player_data = handle_missing_values(player_data)

    # Agregar os dados dos jogadores, mantendo o GAME_ID
    aggregated_data = player_data.groupby("GAME_ID").agg({
        "PTS": "mean",
        "REB": "mean",
        "AST": "mean",
        "MIN": "mean",
        "PLUS_MINUS": "mean"
    }).reset_index()
    
    # Renomear colunas agregadas
    aggregated_data.columns = ['GAME_ID', 'AVG_PTS', 'AVG_REB', 'AVG_AST', 'AVG_MIN', 'AVG_PLUS_MINUS']
    return aggregated_data

def preprocess_data():
    season_data, player_data = fetch_data_from_db()
    
    # Limpeza dos dados da temporada
    season_data = handle_missing_values(season_data)

    # Normalização dos dados da temporada
    columns_to_normalize_season = ["FG_PCT", "FG3_PCT", "FT_PCT", "REB", "AST", "PTS"]
    season_data = normalize_data(season_data, columns_to_normalize_season)

    # Salvar os dados da temporada em um arquivo
    season_data.to_csv("nba_season_data.csv", index=False)
    print("Dados da temporada pré-processados salvos em nba_season_data.csv")

    # Agregar e processar dados dos jogadores
    aggregated_player_data = aggregate_player_data(player_data)

    # Manter todos os dados dos jogadores e salvar
    player_data_cleaned = handle_missing_values(player_data)  # Limpeza adicional se necessário

    # Salvar os dados agregados dos jogadores em um arquivo
    player_data_cleaned.to_csv("nba_player_data.csv", index=False)
    print("Dados dos jogadores pré-processados salvos em nba_player_data.csv")

if __name__ == "__main__":
    preprocess_data()
