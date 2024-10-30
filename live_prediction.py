import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
import joblib
from nba_api.stats.endpoints import BoxScoreTraditionalV2, BoxScoreAdvancedV2

model = joblib.load("./models/nba_best_model.pkl")

def get_live_game_data(game_id):
    try:
        box_score = BoxScoreTraditionalV2(game_id=game_id)
        player_stats = box_score.player_stats.get_data_frame()
        team_stats = box_score.team_stats.get_data_frame()
        return player_stats, team_stats
    except Exception as e:
        print(f"Erro ao obter dados ao vivo para o game_id {game_id}: {e}")
        return None, None

def preprocess_live_data(player_stats, team_stats):

    team_stats = team_stats[['TEAM_ID', 'PTS', 'PLUS_MINUS']]
    player_stats = player_stats[['TEAM_ID', 'PLAYER_NAME', 'PTS']]
    aggregated_data = player_stats.groupby("TEAM_ID").agg({'PTS': 'mean'}).reset_index()
    aggregated_data.columns = ['TEAM_ID', 'AVG_PTS']

    merged_data = team_stats.merge(aggregated_data, on="TEAM_ID", how="left")
    scaler = StandardScaler()
    return scaler.fit_transform(merged_data[['PTS', 'PLUS_MINUS', 'AVG_PTS']])

def initial_prediction(player_stats, team_stats):
    data = preprocess_live_data(player_stats, team_stats)
    
    team1_name = team_stats.iloc[0]['TEAM_NAME']
    team2_name = team_stats.iloc[1]['TEAM_NAME']
    
    initial_prediction = model.predict(data)
    win_prob = model.predict_proba(data)
    
    predicted_winner = team1_name if initial_prediction[0] == 1 else team2_name
    print(f"Predição Inicial: Time vencedor esperado - {predicted_winner}")
    print(f"Probabilidade de Vitória: {win_prob[0][1] * 100:.2f}% para {team1_name} | {win_prob[0][0] * 100:.2f}% para {team2_name}")

    #Prever os 5 maiores pontuadores do jogo
    predicted_top_scorers = player_stats.nlargest(5, 'PTS')[['PLAYER_NAME', 'PTS']]
    print("\nPrevisão dos Maiores Pontuadores do Jogo:")
    print(predicted_top_scorers)
    
    #Estimar o placar final previsto
    predicted_score_team1 = int(team_stats.loc[0, 'PTS'] + (team_stats.loc[0, 'PLUS_MINUS'] / 2))
    predicted_score_team2 = int(team_stats.loc[1, 'PTS'] - (team_stats.loc[1, 'PLUS_MINUS'] / 2))
    print(f"\nPlacar Inicial Previsto - {team1_name}: {predicted_score_team1} | {team2_name}: {predicted_score_team2}")
    
    return predicted_score_team1, predicted_score_team2, predicted_top_scorers, team1_name, team2_name

#Loop para atualizar a previsão durante o jogo
def update_predictions(game_id, predicted_score_team1, predicted_score_team2, team1_name, team2_name):
    while True:
        player_stats, team_stats = get_live_game_data(game_id)
        if player_stats is None or team_stats is None:
            break

        #Recalcular a chance de vitória
        updated_data = preprocess_live_data(player_stats, team_stats)
        win_prob = model.predict_proba(updated_data)
        
        #Exibir as informações atualizadas no console
        print("\n--- Atualização ---")
        print(f"Placar Previsto Inicial - {team1_name}: {predicted_score_team1} | {team2_name}: {predicted_score_team2}")
        print(f"Placar Atual - {team1_name}: {team_stats.iloc[0]['PTS']} | {team2_name}: {team_stats.iloc[1]['PTS']}")
        print(f"Probabilidade de Vitória Atualizada: {win_prob[0][1] * 100:.2f}% para {team1_name} | {win_prob[0][0] * 100:.2f}% para {team2_name}")
        
        #Pausar antes da próxima atualização
        time.sleep(60)  # Atualiza a cada minuto

def main():
    game_id = input("Digite o ID do jogo ao vivo: ")
    player_stats, team_stats = get_live_game_data(game_id)

    if player_stats is not None and team_stats is not None:
        #Fazer a previsão inicial
        predicted_score_team1, predicted_score_team2, predicted_top_scorers, team1_name, team2_name = initial_prediction(player_stats, team_stats)
        
        #Atualizar previsões durante o jogo
        update_predictions(game_id, predicted_score_team1, predicted_score_team2, team1_name, team2_name)
    else:
        print("Não foi possível obter dados ao vivo para o jogo.")

if __name__ == "__main__":
    main()
