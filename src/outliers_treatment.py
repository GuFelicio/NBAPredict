import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Função para carregar dados de um arquivo CSV
def load_data(filepath):
    return pd.read_csv(filepath)

# Função para gerar boxplots de variáveis selecionadas
def plot_boxplots(data, columns):
    for column in columns:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data[column])
        plt.title(f"Boxplot de {column}")
        plt.show()

# Função para gerar scatter plots
def plot_scatter(data, x_col, y_col):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data[x_col], y=data[y_col])
    plt.title(f"Scatter Plot de {x_col} vs {y_col}")
    plt.show()

# Função para gerar histogramas
def plot_histograms(data, columns):
    for column in columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(data[column], kde=True)
        plt.title(f"Histograma de {column}")
        plt.show()

# Função principal para visualizar outliers
def visualize_outliers(data):
    numeric_columns = ["FG_PCT", "FG3_PCT", "FT_PCT", "REB", "AST", "PTS"]

    print("Gerando boxplots para visualização de outliers...")
    plot_boxplots(data, numeric_columns)

    print("Gerando histogramas para distribuição...")
    plot_histograms(data, numeric_columns)

    if 'AVG_PTS' in data.columns:  # Verifica se a coluna AVG_PTS existe
        print("Gerando scatter plot para PTS vs AVG_PTS...")
        plot_scatter(data, "PTS", "AVG_PTS")

# Função para tratar outliers usando o método do IQR
def treat_outliers(data, columns):
    for column in columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        data = data[~((data[column] < (Q1 - 1.5 * IQR)) | (data[column] > (Q3 + 1.5 * IQR)))]
    return data

def main():
    #Carregar os dados dos dois arquivos
    season_data = load_data("nba_season_data.csv")
    player_data = load_data("nba_player_data.csv")

    #Visualizar e tratar outliers nos dados da temporada
    print("Tratando outliers nos dados da temporada...")
    visualize_outliers(season_data)
    columns_to_treat_season = ["FG_PCT", "FG3_PCT", "FT_PCT", "REB", "AST", "PTS"]
    season_data_cleaned = treat_outliers(season_data, columns_to_treat_season)
    season_data_cleaned.to_csv("nba_season_data_no_outliers.csv", index=False)
    print("Dados da temporada tratados e salvos em nba_season_data_no_outliers.csv")

    #Visualizar e tratar outliers nos dados dos jogadores
    print("Tratando outliers nos dados dos jogadores...")
    visualize_outliers(player_data)
    columns_to_treat_player = ["PTS", "REB", "AST", "MIN", "PLUS_MINUS"]
    player_data_cleaned = treat_outliers(player_data, columns_to_treat_player)
    player_data_cleaned.to_csv("nba_player_data_no_outliers.csv", index=False)
    print("Dados dos jogadores tratados e salvos em nba_player_data_no_outliers.csv")

if __name__ == "__main__":
    main()
