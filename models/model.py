
from src.data.acquisition import fetch_data
from src.data.preprocessing import process_data, prepare_models_and_split, create_processed_file, impute_missing_values
from src.data.train_test import train_and_evaluate, show_results
import pandas as pd

def run_pipeline():
    # Etapa 1: Baixar e descompactar os dados.
    fetch_data("https://static.nhtsa.gov/odi/ffdd/cmpl/COMPLAINTS_RECEIVED_2015-2019.zip")
    
    # Etapa 2: Criar o arquivo .csv processado, nomeando as colunas corretamente segundo o arquivo original,
    # dropando as colunas desnecessárias para a tarefa planejada, 
    # fazendo limpeza dos textos, normalização nas colunas númericas, transformações nas colunas binárias,
    # aplicando OneHotEncoding nas colunas categóricas, aplicando o TfidVectorizer para criar 50 novas features
    # com base nas colunas de texto e por fim salvando as alterações no .csv gerado.
    create_processed_file()


    # Etapa 3: Lendo o DataFrame para fazer as últimas alterações, preenchendo os valores nulos usando o SimpleImputer. 
    input_file = "models/data/processed/complaints.csv"
    df = pd.read_csv(input_file)
    df_imputado = impute_missing_values(df)
    df_imputado.to_csv(input_file)

    # Etapa 3: Separando os dados em treino, teste, convertendo o target de volta para binário, setando os models
    # e iniciando o treinamento e teste dos modelos.
    models, X_train, X_test, y_train, y_test = prepare_models_and_split(df_imputado)
    results = train_and_evaluate(models, X_train, y_train, X_test, y_test)

    # Etapa 4: Mostrar todos os resultados dos modelos treinados usando as métricas: Accuracy, Precision, Recall e F1-Score
    show_results(results)
# Executando o pipeline
run_pipeline()