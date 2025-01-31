from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import os
import joblib
from src.data.preprocessing import prepare_models_and_split, process_data

# Função para treinar e avaliar os modelos
def train_and_evaluate(models, X_train, y_train, X_test, y_test):
    """
    Função para treinar e avaliar todos os modelos selecionados usando os dados que forem passados.

    Parâmetros:
    models: Dicionário com os modelos treinados.
    X_train, X_test, y_train, y_test: Conjuntos de dados divididos para treino e teste.

    Retorna:
    results: Dicionário com todos os resultados de todos modelos que foram treinados e testados.
    """

    results = {}
    
    # Barra de progresso para o treinamento de todos os modelos
    total_steps = len(models)  # Número de modelos
    with tqdm(total=total_steps, desc="Treinando modelos", unit="modelo") as pbar:
        for name, model in models.items():
            # Treinamento do modelo
            model.fit(X_train, y_train)
            
            # Previsão e cálculo das métricas
            y_pred = model.predict(X_test)
            
            results[name] = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average='weighted'),
                "Recall": recall_score(y_test, y_pred, average='weighted'),
                "F1-score": f1_score(y_test, y_pred, average='weighted')
            }
            
            pbar.set_postfix(model=name + " concluído")  # Atualiza a barra com o nome do modelo concluído
            pbar.update(1)  # Atualiza o progresso
    
    return results

def show_results(results):
    """
    Função para exibir os resultados de avaliação dos modelos.

    Parâmetros:
    results: Dicionário com os resultados, onde a chave é o nome do modelo 
             e o valor é outro dicionário com as métricas e seus valores.
    """
    for model, metrics in results.items():
        print(f"\n{model}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

def save_models(models, output_dir="models"):
    """
    Função que salva todos os modelos treinados em .pkl

    Parâmetros:
    models: dicionário com os modelos treinados
    output_dir: string com o path do diretório onde os modelos serão salvos.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name, model in models.items():
        # Caminho para salvar o modelo
        model_file = os.path.join(output_dir, f"{model_name.replace(' ', '_')}.pkl")
        
        try:
            # Salvando o modelo
            joblib.dump(model, model_file)
            print(f"Modelo {model_name} salvo em: {model_file}")
        except Exception as e:
            print(f"Erro ao salvar o modelo {model_name}: {e}")

# Função para carregar os modelos
def load_model(model_name, input_dir="models"):
    """
    Função que carrega os modelos salvos para serem utilizados.

    Parâmetros:
    model_name: Nome(s) do modelo que deseja ser carregado para utilização.
    input_dir: string com path para o diretório onde os modelos serão encontrados.

    Retorna:
    model: Modelo carregado e pronto para utilização.
    """
    
    model_file = os.path.join(input_dir, f"{model_name.replace(' ', '_')}.pkl")
    
    try:
        model = joblib.load(model_file)
        print(f"Modelo {model_name} carregado de: {model_file}")
        return model
    except Exception as e:
        print(f"Erro ao carregar o modelo {model_name}: {e}")
        return None
    

def test_all_models(idx, X=None, y=None, models_dir="models"):
    """
    Testa todos os modelos carregados e compara a previsão com o valor real do target.
    
    Parâmettros:
    idx: Indíce colocado manualmente pelo usuário que vai indicar o elemento do conjunto de dados que vai ser testado.
    X: Recebe por padrão X_test que refere-se aos dados gerais utilizados para teste.
    y: Recebe por padrão y_test que refere-se aos dados do target utilizados para teste.
    models_dir: String contendo o path para o diretório onde se encontram os modelos.

    Retorna:
    results: Dicionário com todos os resultados de todos os modelos que foram testados.
    """
    

    if X is None or y is None:
        raise ValueError("Os parâmetros X e y precisam ser fornecidos.")

    
    # Carregar os modelos usando a função load_model
    models = {}
    model_names = ["Logistic_Regression", "Random_Forest", "XGBoost"]  # Nomes dos modelos
    
    for model_name in model_names:
        model = load_model(model_name, models_dir)
        if model is not None:
            models[model_name] = model
    
    # Selecionar o exemplo do DataFrame
    X_single = X.iloc[[idx]]  # A linha escolhida
    y_single = y.iloc[idx]  # O valor real do target para esse exemplo
    
    # Dicionário para armazenar as previsões e acertos
    results = {}
    
    # Iterar sobre os modelos e fazer as previsões
    for name, model in models.items():
        try:
            # Realizando a previsão para o exemplo escolhido
            y_pred = model.predict(X_single)
            
            # Verificando se a previsão foi correta
            is_correct = y_pred[0] == y_single
            
            # Armazenando a informação
            results[name] = {
                "Predição": y_pred[0],
                "Valor Real": y_single,
                "Acerto": is_correct
            }
        except Exception as e:
            # Caso ocorra algum erro, armazena a mensagem de erro
            results[name] = {"Erro": str(e)}
    
    return results