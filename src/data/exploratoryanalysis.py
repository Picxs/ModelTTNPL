import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.data.train_test import load_model
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

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

    # Aqui exibe as primeiras linhas
    print("\nPrimeiras linhas do dataset:")
    print(df.head())

    # Aqui é mostrado as informações gerais
    print("\nInformações gerais do dataset:")
    print(df.info())

    # Aqui é exibe as estatísticas descritivas
    print("\nEstatísticas descritivas:")
    print(df.describe(include=['number']))

    # Aqui é mostrado a soma dos valores ausentes por coluna
    # É liberado também o limite de linhas mostradas para que a análise de valores nulos seja precisa.
    pd.set_option('display.max_rows', None)
    print("\nValores ausentes por coluna:")
    print(df.isnull().sum())



def plot_valid_counts(input_file):
    """
    Gera um gráfico de barras mostrando a quantidade de valores válidos (não nulos) em cada coluna.
    
    Parâmetros:
    input_file: string contendo o path para o arquivo de dados .csv que vai ser utilizado para plotar o gráfico.
    """
    if not os.path.exists(input_file):
        print(f"Erro: O arquivo '{input_file}' não foi encontrado.")
        return

    try:
        df = pd.read_csv(input_file)
        print("Arquivo carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return

    # Aqui é contado os valores válidos (não nulos) em cada coluna.
    valid_counts = df.notnull().sum()

    # Aqui é plotado o gráfico usando o matplotlib.
    plt.figure(figsize=(12, 6))
    valid_counts.sort_values(ascending=False).plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title("Quantidade de Valores Válidos por Coluna", fontsize=16)
    plt.xlabel("Colunas", fontsize=12)
    plt.ylabel("Quantidade de Valores Válidos", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()




def show_unique_values(input_file):
    """
    Mostra os valores únicos de cada variável.
    
    Parâmetros:
    input_file: string contendo o path para o arquivo de dados .csv que vai ser utilizado para verificar os valores únicos de cada
    variável.
    """

    if not os.path.exists(input_file):
        print(f"Erro: O arquivo '{input_file}' não foi encontrado.")
        return

    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")
        return

    # Aqui é selecionado apenas as colunas numéricas
    numeric_columns = df.select_dtypes(include=['number']).columns

    pd.set_option('display.max_rows', 10)

    # Aqui mostra os valores únicos de cada variável numérica
    for col in numeric_columns:
        print(f"\nValores únicos em {col}:")
        unique_values = df[col].value_counts().sort_index()
        print(unique_values)

    print("\nValores únicos de todas as variáveis numéricas foram exibidos.")

def plot_feature_importance(models_dir="models", X_train=None):
    """
    Plota a importância das features para todos os modelos carregados, mostrando apenas as 10 features mais importantes.

    Parâmetros:
    models_dir: Diretório onde os modelos estão armazenados.
    X_train: Dados de treinamento.
    """

    if X_train is None:
        raise ValueError("Os parâmetros X e y precisam ser fornecidos.")
    
    models = {}
    model_names = ["Logistic_Regression", "Random_Forest", "XGBoost"]

    for model_name in model_names:
        model = load_model(model_name, models_dir)
        if model is not None:
            models[model_name] = model
    
    results = {}

    for name, model in models.items():
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = X_train.columns

                # Seleciona as 10 features mais importantes
                top_10_indices = importances.argsort()[-10:][::-1] 
                top_10_importances = importances[top_10_indices]   
                top_10_feature_names = feature_names[top_10_indices]  

                # Plota o gráfico
                plt.figure(figsize=(10, 7))
                plt.barh(top_10_feature_names, top_10_importances)
                plt.xlabel('Importância')
                plt.title(f'Top 10 Features Mais Importantes - {name}')
                plt.show()
            else:
                print(f"{name} não possui importância das features.")
        
        except Exception as e:
            results[name] = {"Erro": str(e)}
    
    return results


def plot_confusion_matrix(X=None, y=None, models_dir="models"):
    """
    Plota a matriz de confusão para todos os modelos carregados e exibe os valores de TP, FN, FP, TN de cada um.
    
    Parâmetros:
    X: Dados de entrada (padrão: X_test).
    y: Dados de saída (padrão: y_test).
    models_dir: Diretório onde os modelos estão salvos.
    
    Retorna:
    None: Exibe a matriz de confusão e os valores de TP, FN, FP, TN de cada modelo.
    """

    if X is None or y is None:
        raise ValueError("Os parâmetros X e y precisam ser fornecidos.")

    models = {}
    model_names = ["Logistic_Regression", "Random_Forest", "XGBoost"] 
    
    for model_name in model_names:
        model = load_model(model_name, models_dir)
        if model is not None:
            models[model_name] = model

    for name, model in models.items():
        # Aqui são feitas as previsões para o modelo
        y_pred = model.predict(X)
        
        # Aqui é calculado a matriz de confusão
        cm = confusion_matrix(y, y_pred)
        
        # Extraindo TP, FN, FP, TN da matriz de confusão
        tp, fn, fp, tn = cm.ravel()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'])
        plt.title(f'Matriz de Confusão - {name}')
        plt.xlabel('Predição')
        plt.ylabel('Real')
        plt.show()
        
        print(f"Modelo: {name}")
        print(f"TP (Verdadeiro Positivo): {tp}")
        print(f"FN (Falso Negativo): {fn}")
        print(f"FP (Falso Positivo): {fp}")
        print(f"TN (Verdadeiro Negativo): {tn}")
        print("-" * 40)


def plot_roc_curve(models_dir="models", X=None, y=None):
    """
    Plota a Curva ROC para todos os modelos carregados.

    Parâmetros:
    models_dir: Diretório onde os modelos estão armazenados.
    X: Dados de teste.
    y: Valores reais do target.
    """

    if X is None or y is None:
        raise ValueError("Os parâmetros X e y precisam ser fornecidos.")

    models = {}
    model_names = ["Logistic_Regression", "Random_Forest", "XGBoost"]

    for model_name in model_names:
        model = load_model(model_name, models_dir)
        if model is not None:
            models[model_name] = model
    
    results = {}

    for name, model in models.items():
        try:
            # Aqui é calculado as probabilidades para usar na ROC
            y_prob = model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, y_prob)
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Curva ROC - {name}')
            plt.legend(loc='lower right')
            plt.show()
        
        except Exception as e:
            results[name] = {"Erro": str(e)}
    
    return results



def plot_precision_recall_curve(models_dir="models", X=None, y=None):
    """
    Plota a Curva de Precisão-Revocação para todos os modelos carregados.

    Parâmetros:
    models_dir: Diretório onde os modelos estão armazenados.
    X: Dados de teste.
    y: Valores reais do target.
    """

    if X is None or y is None:
        raise ValueError("Os parâmetros X e y precisam ser fornecidos.")
    
    models = {}
    model_names = ["Logistic_Regression", "Random_Forest", "XGBoost"]

    for model_name in model_names:
        model = load_model(model_name, models_dir)
        if model is not None:
            models[model_name] = model
    
    results = {}

    for name, model in models.items():
        try:
            y_prob = model.predict_proba(X)[:, 1]  # Para classificação binária
            precision, recall, _ = precision_recall_curve(y, y_prob)

            plt.figure()
            plt.plot(recall, precision, color='blue', lw=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Curva de Precisão-Revocação - {name}')
            plt.show()
        
        except Exception as e:
            results[name] = {"Erro": str(e)}
    
    return results