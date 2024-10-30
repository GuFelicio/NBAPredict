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
def load_data(filepath="./data/nba_detailed_merged_data.csv"):
    return pd.read_csv(filepath)

def split_data(data):
    #Converter a coluna 'WL' para valores binários (1 para vitória, 0 para derrota)
    data["WL"] = data["WL"].apply(lambda x: 1 if x == 'W' else 0)
    X = data.drop(columns=["WL"])  
    y = data["WL"]  

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    column_transformer = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_cols)  
        ],
        remainder='passthrough'  
    )

    X_transformed = column_transformer.fit_transform(X)
    X_transformed = pd.DataFrame(X_transformed.toarray())

    # Tratar valores ausentes
    if X_transformed.isnull().any().any():  
        X_transformed = X_transformed.fillna(0) 
    return train_test_split(X_transformed, y, test_size=0.2, random_state=42)

def create_ensemble_model():
    rf = RandomForestClassifier(random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    ensemble = VotingClassifier(
        estimators=[("RandomForest", rf), ("GradientBoosting", gb)],
        voting="soft"  # "soft" considera as probabilidades para maior precisão
    )
    return ensemble

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

    # Salvar o modelo ensemble
    joblib.dump(ensemble_model, "./models/nba_ensemble_model.pkl")
    print("Modelo ensemble salvo como nba_ensemble_model.pkl")

if __name__ == "__main__":
    main()
