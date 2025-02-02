from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import os
import joblib
from src.data.preprocessing import prepare_models_and_split, process_data, roc_auc_score
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

def train_and_evaluate(models, X_train, y_train, X_test, y_test):
    """
    Função para treinar e avaliar todos os modelos selecionados, incluindo um Voting Classifier.

    Parâmetros:
    models: Dicionário com os modelos individuais.
    X_train, X_test, y_train, y_test: Conjuntos de dados divididos para treino e teste.

    Retorna:
    results: Dicionário com os resultados de todos os modelos e do Voting Classifier.
    """

    results = {}

    # Aqui é feito o treino dos modelos.
    total_steps = len(models) + 1  # +1 para o Voting Classifier
    with tqdm(total=total_steps, desc="Treinando modelos", unit="modelo") as pbar:
        for name, model in models.items():
            # Aqui temos o treinamento do modelo.
            model.fit(X_train, y_train)

            # Aqui é feita a prrevisão e cálculo das métricas.
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            # Aqui é a listagem de todas as métricas que serão utilizadas pelos 3 modelos tradicionais.
            results[name] = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average='weighted'),
                "Recall": recall_score(y_test, y_pred, average='weighted'),
                "F1-score": f1_score(y_test, y_pred, average='weighted'),
            }

            # Aqui fazemos o calcúlo do AUC-ROC
            if y_prob is not None:
                results[name]["AUC-ROC"] = roc_auc_score(y_test, y_prob)

            pbar.set_postfix(model=name + " concluído")
            pbar.update(1)

        models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        # Aqui é criado o Voting Classifier usando os modelos.
        voting_clf = VotingClassifier(
            estimators=[(name, model) for name, model in models.items()],
            voting='soft',
            n_jobs=-1
        )

        # Aqui é o treino do Voting Classifier.
        voting_clf.fit(X_train, y_train)

        # Aqui são as previsões e cálculos das métricas para o ensemble.
        y_pred_voting = voting_clf.predict(X_test)
        y_prob_voting = voting_clf.predict_proba(X_test)[:, 1]

        # Aqui temos a lista de todas as métricas que serão utilizadas pelo Voting Classifier
        results["Voting Classifier"] = {
            "Accuracy": accuracy_score(y_test, y_pred_voting),
            "Precision": precision_score(y_test, y_pred_voting, average='weighted'),
            "Recall": recall_score(y_test, y_pred_voting, average='weighted'),
            "F1-score": f1_score(y_test, y_pred_voting, average='weighted'),
            "AUC-ROC": roc_auc_score(y_test, y_prob_voting)
        }

        pbar.set_postfix(model="Voting Classifier concluído")
        pbar.update(1)

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
        model_file = os.path.join(output_dir, f"{model_name.replace(' ', '_')}.pkl")
        
        try:
            # Aqui usamos o joblib para salvar o modelo
            joblib.dump(model, model_file)
            print(f"Modelo {model_name} salvo em: {model_file}")
        except Exception as e:
            print(f"Erro ao salvar o modelo {model_name}: {e}")


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
        # Aqui usamos novamente o joblib, mas agora para carregar o modelo
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
    
    if idx < 0 or idx >= len(X):
        return {"Erro": f"Índice {idx} fora do intervalo válido. O valor de idx deve ser entre 0 e {len(X)-1}"}
    
    # Aqui carregamos os modelos usando a função load_model
    models = {}
    model_names = ["Logistic_Regression", "Random_Forest", "XGBoost"] 
    
    for model_name in model_names:
        model = load_model(model_name, models_dir)
        if model is not None:
            models[model_name] = model
    
    X_single = X.iloc[[idx]]  
    y_single = y.iloc[idx]  # O valor real do target para esse exemplo
    
    
    results = {}
    
    # Aqui nos percorremos sobre os modelos e fazemos as previsões
    for name, model in models.items():
        try:
            y_pred = model.predict(X_single)
            
            # Verificamos se a previsão foi correta
            is_correct = y_pred[0] == y_single
            
            results[name] = {
                "Predição": y_pred[0],
                "Valor Real": y_single,
                "Acerto": is_correct
            }
        except Exception as e:
            results[name] = {"Erro": str(e)}
    
    return results


def plot_learning_curve(models_dir="models", X_train=None, y_train=None):
    """
    Plota a Learning Curve para todos os modelos carregados.

    Parâmetros:
    models_dir: Diretório onde os modelos estão armazenados.
    X_train: Dados de treinamento.
    y_train: Valores reais do target de treinamento.
    """

    if X_train is None or y_train is None:
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
            train_sizes, train_scores, val_scores = learning_curve(model, X_train, y_train, n_jobs=-1)
            train_mean = train_scores.mean(axis=1)
            val_mean = val_scores.mean(axis=1)

            plt.figure(figsize=(10, 7))
            plt.plot(train_sizes, train_mean, label='Erro de Treinamento', color='blue')
            plt.plot(train_sizes, val_mean, label='Erro de Validação', color='red')
            plt.xlabel('Número de Exemplos de Treinamento')
            plt.ylabel('Erro')
            plt.title(f'Curva de Aprendizado - {name}')
            plt.legend()
            plt.show()
        
        except Exception as e:
            results[name] = {"Erro": str(e)}
    
    return results