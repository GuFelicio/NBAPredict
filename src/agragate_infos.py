import pandas as pd

def load_data(season_file, player_file):
    season_data = pd.read_csv(season_file)
    player_data = pd.read_csv(player_file)
    return season_data, player_data

def aggregate_player_data(player_data):
    # Preencher valores ausentes com 0
    player_data.fillna(0, inplace=True)

    # Agregar dados dos jogadores por GAME_ID mantendo o PLAYER_ID
    aggregated_data = player_data.groupby("GAME_ID").agg({
        "PLAYER_ID": "first",  # Mantém o primeiro PLAYER_ID encontrado
        "PTS": "mean",
        "REB": "mean",
        "AST": "mean",
        "MIN": "mean",
        "PLUS_MINUS": "mean"
    }).reset_index()
    
    # Renomear colunas
    aggregated_data.columns = ['GAME_ID', 'PLAYER_ID', 'AVG_PTS', 'AVG_REB', 'AVG_AST', 'AVG_MIN', 'AVG_PLUS_MINUS']
    return aggregated_data

def merge_data(season_data, aggregated_player_data):
    # Mesclar os dados dos jogos com os dados agregados dos jogadores
    merged_data = pd.merge(season_data, aggregated_player_data, on="GAME_ID", how="left")
    return merged_data

def create_new_features(data):
    # Criar novas features
    data['PTS_EFFICIENCY'] = data['PTS'] / data['AVG_PTS'].replace(0, 1)  # Para evitar divisão por zero
    data['PTS_DIFF'] = data['PTS'] - data['AVG_PTS']
    
    # Adicionando uma feature que indica se o time jogou em casa ou fora
    data['HOME_TEAM'] = data['MATCHUP'].apply(lambda x: 1 if 'vs' in x else 0)
    
    return data

def save_final_data(data, output_file):
    data.to_csv(output_file, index=False)
    print(f"Dados mesclados e tratados salvos em {output_file}")

def main():
    season_file = "nba_season_data_no_outliers.csv"
    player_file = "nba_player_data_no_outliers.csv"
    output_file = "nba_final_merged_data.csv"
    
    # Carregar dados
    season_data, player_data = load_data(season_file, player_file)

    # Agregar dados dos jogadores
    aggregated_player_data = aggregate_player_data(player_data)

    # Mesclar dados
    merged_data = merge_data(season_data, aggregated_player_data)

    # Criar novas features
    final_data = create_new_features(merged_data)

    # Salvar o conjunto de dados final
    save_final_data(final_data, output_file)

if __name__ == "__main__":
    main()
