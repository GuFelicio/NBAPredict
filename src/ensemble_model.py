# ensemble_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

#Carregar dados pré-processados sem outliers
def load_data(filepath="nba_final_merged_data.csv"):
    return pd.read_csv(filepath)

#Função para dividir dados em treino e teste
def split_data(data):
    # Separar recursos e variável alvo
    X = data.drop(columns=["WL"])  # Recursos
    y = data["WL"]  # Variável alvo

    # Identificar colunas categóricas
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Criar um transformador para codificar colunas categóricas
    column_transformer = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_cols)  # Aplica OneHotEncoder nas colunas categóricas
        ],
        remainder='passthrough'  # Mantém as outras colunas
    )

    # Aplicar a transformação
    X_transformed = column_transformer.fit_transform(X)

    # Converter matriz esparsa para um DataFrame denso
    X_transformed = pd.DataFrame(X_transformed.toarray())  # Converte csr_matrix para array

    # Tratar valores ausentes
    if X_transformed.isnull().any().any():  # Verificar se há NaN
        X_transformed = X_transformed.fillna(0)  # Preencher NaNs com 0

    # Dividir os dados em conjuntos de treino e teste
    return train_test_split(X_transformed, y, test_size=0.2, random_state=42)
#Função para criar o modelo ensemble
def create_ensemble_model():
    rf = RandomForestClassifier(random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    mlp = MLPClassifier(random_state=42, max_iter=300)
    ensemble = VotingClassifier(
        estimators=[("RandomForest", rf), ("GradientBoosting", gb), ("MLP", mlp)],
        voting="soft"  #"soft" considera as probabilidades para maior precisão
    )
    return ensemble

#Função principal para treinar e avaliar o ensemble model
def main():
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)

    ensemble_model = create_ensemble_model()
    ensemble_model.fit(X_train, y_train)

    y_pred = ensemble_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Acurácia do Ensemble: {accuracy:.4f}")
    print(f"F1 Score do Ensemble: {f1:.4f}")
    print("\nRelatório de Classificação do Ensemble:\n", classification_report(y_test, y_pred))

    #Salvar o modelo ensemble
    joblib.dump(ensemble_model, "nba_ensemble_model.pkl")
    print("Modelo ensemble salvo como nba_ensemble_model.pkl")

if __name__ == "__main__":
    main()
