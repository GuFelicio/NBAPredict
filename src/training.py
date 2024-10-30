import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

#Carregar dados pré-processados sem outliers
def load_data(filepath="./data/nba_detailed_merged_data.csv"):
    return pd.read_csv(filepath)

#Função para dividir dados em treino e teste
def split_data(data):
    data["WL"] = data["WL"].apply(lambda x: 1 if x == 'W' else 0)

    X = data.drop(columns=["WL"]) 
    y = data["WL"]

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    column_transformer = ColumnTransformer(
        transformers=[
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Preenche NaNs em categóricas
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_cols)
        ],
        remainder='passthrough'
    )

    X_transformed = column_transformer.fit_transform(X)

    #Converter matriz esparsa para um DataFrame denso e tratar valores ausentes
    X_transformed = pd.DataFrame(X_transformed.toarray()).fillna(0)  
    return train_test_split(X_transformed, y, test_size=0.2, random_state=42)

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "MLP": MLPClassifier(random_state=42, max_iter=300)
    }
    best_model = None
    best_accuracy = 0

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n{model_name} - Acurácia: {accuracy:.4f}, F1 Score: {f1:.4f}")
        print(classification_report(y_test, y_pred))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = model_name

    print(f"\nMelhor modelo: {best_model_name} com Acurácia de {best_accuracy:.4f}")
    return best_model

def main():
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)

    best_model = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    joblib.dump(best_model, "./models/nba_best_model.pkl")
    print("Melhor modelo salvo como nba_best_model.pkl")

if __name__ == "__main__":
    main()
