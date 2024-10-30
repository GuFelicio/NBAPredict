import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

# Carregar dados pré-processados sem outliers
def load_data(filepath="./data/nba_detailed_merged_data.csv"):
    return pd.read_csv(filepath)

# Função para preparar os dados com OneHotEncoding e divisão de variáveis
def prepare_data(data):
    # Separar recursos e variável alvo
    X = data.drop(columns=["WL"])  # Recursos
    y = data["WL"].fillna("L")  # Variável alvo com valores ausentes tratados

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

    # Tratar valores ausentes nos recursos
    X_transformed = X_transformed.fillna(0)

    return X_transformed, y

# Função para realizar a validação cruzada nos modelos, incluindo o ensemble
def cross_validate_models(X, y):
    rf = RandomForestClassifier(random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    ensemble = VotingClassifier(estimators=[("RandomForest", rf), ("GradientBoosting", gb)], voting="soft")

    models = {
        "RandomForest": rf,
        "GradientBoosting": gb,
        "Ensemble (RF + GB)": ensemble
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for model_name, model in models.items():
        print(f"Validando modelo {model_name}...")

        # Realizar a validação cruzada
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        
        # Exibir a média e o desvio padrão da acurácia do modelo
        print(f"{model_name} - Acurácia média: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        results[model_name] = cv_scores

    return results

# Função principal para carregar, preparar dados e realizar a validação cruzada
def main():
    data = load_data()
    X, y = prepare_data(data)

    results = cross_validate_models(X, y)

    # Exibir a média e desvio padrão de cada modelo
    for model_name, scores in results.items():
        print(f"\n{model_name} - Acurácia média em 5-fold CV: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

if __name__ == "__main__":
    main()
