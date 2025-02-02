
from src.data.acquisition import fetch_data
from src.data.preprocessing import process_data, prepare_models_and_split, create_processed_file, impute_missing_values, clean_text, removing_nan
from src.data.train_test import train_and_evaluate, show_results, save_models
from src.data.exploratoryanalysis import exploratory_analysis
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


    # Etapa 3: Lendo o DataFrame para fazer a primeira análise exploratória e últimas alterações, preenchendo os valores nulos usando 
    # o SimpleImputer e removendo as colunas _nan. 
    input_file = "data/processed/complaints.csv"
    exploratory_analysis(input_file)

    df = pd.read_csv(input_file)
    df_imputado = impute_missing_values(df)

    df_completo = removing_nan(df_imputado)
    df_completo.to_csv("data/processed/complaints.csv", index=False)

    exploratory_analysis(input_file)

    # Etapa 4: Separando os dados em treino, teste, convertendo o target de volta para binário, setando os models
    # e iniciando o treinamento e teste dos modelos.
    df_treino = pd.read_csv("data/processed/complaints.csv")
    models, X_train, X_test, y_train, y_test = prepare_models_and_split(df_treino)
    results = train_and_evaluate(models, X_train, y_train, X_test, y_test)

    # Etapa 5: Mostrar todos os resultados dos modelos treinados usando as métricas: Accuracy, Precision, Recall e F1-Score e salvar os modelos 
    # resutantes do treinamento.
    show_results(results)

    save_models(models)


# Executando o pipeline
run_pipeline()