import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_valid_counts(input_file):
    """
    Gera um gráfico de barras mostrando a quantidade de valores válidos (não nulos) em cada coluna.
    
    Parâmetros:
    input_file: string contendo o path para o arquivo de dados .csv que vai ser utilizado para plotar o gráfico.
    """
    # Verificar se o arquivo existe
    if not os.path.exists(input_file):
        print(f"Erro: O arquivo '{input_file}' não foi encontrado.")
        return

    # Carregar o dataset
    try:
        df = pd.read_csv(input_file)
        print("Arquivo carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return

    # Contar valores válidos (não nulos) em cada coluna
    valid_counts = df.notnull().sum()

    # Criar o gráfico
    plt.figure(figsize=(12, 6))
    valid_counts.sort_values(ascending=False).plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title("Quantidade de Valores Válidos por Coluna", fontsize=16)
    plt.xlabel("Colunas", fontsize=12)
    plt.ylabel("Quantidade de Valores Válidos", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Mostrar o gráfico
    plt.show()

def exploratory_analysis(input_file):
    """
    Realiza uma análise exploratória básica dos dados.
    
    Parâmetros:
    input_file: string contendo o path para o arquivo de dados .csv que vai ser utilizado para fazer a análise exploratória.
    """
    if not os.path.exists(input_file):
        print(f"Erro: O arquivo '{input_file}' não foi encontrado.")
        return

    try:
        df = pd.read_csv(input_file, low_memory=False)
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return

    # Exibir as primeiras linhas
    print("\nPrimeiras linhas do dataset:")
    print(df.head())

    # Informações gerais
    print("\nInformações gerais do dataset:")
    print(df.info())

    # Estatísticas descritivas
    print("\nEstatísticas descritivas:")
    print(df.describe(include=['number']))

    # Verificar valores ausentes
    print("\nValores ausentes por coluna:")
    print(df.isnull().sum())

    # Análise de distribuições
    print("\nDistribuições das variáveis categóricas:")
    for col in df.select_dtypes(include=['object', 'bool']):
        print(f"\n{col} - Distribuição de valores:")
        print(df[col].value_counts())



def show_unique_values(input_file):
    """
    Mostra os valores únicos de cada variável.
    
    Parâmetros:
    input_file: string contendo o path para o arquivo de dados .csv que vai ser utilizado para verificar os valores únicos de cada
    variável.
    """
    # Verificar se o arquivo existe
    if not os.path.exists(input_file):
        print(f"Erro: O arquivo '{input_file}' não foi encontrado.")
        return

    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return

    # Selecionar apenas as colunas numéricas
    numeric_columns = df.select_dtypes(include=['number']).columns

    # Mostrar os valores únicos de cada variável numérica
    for col in numeric_columns:
        print(f"\nValores únicos em {col}:")
        unique_values = df[col].value_counts().sort_index()  # Contagem dos valores únicos
        print(unique_values)

    print("\nValores únicos de todas as variáveis numéricas foram exibidos.")


