from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.data.train_test import test_all_models, load_model  # Adapte conforme o caminho correto para a função
from sklearn.model_selection import train_test_split

app = FastAPI()

# Alterando a função para usar na api
def test_all_models(idx, models_dir="models"):
    """
    Testa todos os modelos carregados e compara a previsão com o valor real do target.
    
    Parâmetros:
    idx: Indíce colocado manualmente pelo usuário que vai indicar o elemento do conjunto de dados que vai ser testado.
    models_dir: String contendo o path para o diretório onde se encontram os modelos.

    Retorna:
    results: Dicionário com todos os resultados de todos os modelos que foram testados.
    """
    
    # Carregar os dados de teste
    try:
        df = pd.read_csv("notebooks/data/processed/complaints.csv")
        y = df["MEDICAL_ATTN"]
        X = df.drop(columns=["MEDICAL_ATTN"])

        # Aqui é feita a divisão de treino e teste, separando 20% dos dados para o teste.
        test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    except Exception as e:
        raise ValueError(f"Erro ao carregar os dados: {str(e)}")
    
    # Verifique se o índice é válido
    if idx < 0 or idx >= len(X_test):
        return {"Erro": f"Índice {idx} fora do intervalo válido. O valor de idx deve ser entre 0 e {len(X_test)-1}"}
    
    # Carregar os modelos
    models = {}
    model_names = ["Logistic_Regression", "Random_Forest", "XGBoost"] 
    
    for model_name in model_names:
        model = load_model(model_name, models_dir)
        if model is not None:
            models[model_name] = model
    
    X_single = X_test.iloc[[idx]]  # Selecionar a linha de dados para o índice especificado
    y_single = y_test.iloc[idx]  # O valor real do target para esse exemplo
    
    results = {}
    
    # Testar os modelos
    for name, model in models.items():
        try:
            y_pred = model.predict(X_single)
            
            # Verificar se a previsão foi correta
            is_correct = y_pred[0] == y_single
            
            results[name] = {
                "Predição": int(y_pred[0]),
                "Valor Real": y_single,
                "Acerto": bool(is_correct)
            }
        except Exception as e:
            results[name] = {"Erro": str(e)}
    
    return results

app = FastAPI()

# Defina um modelo de dados para receber o índice na API
class PredictionRequest(BaseModel):
    idx: int

@app.post("/predict/")
async def predict(request: PredictionRequest):
    # Chame a função de teste
    try:
        results = test_all_models(request.idx, models_dir="notebooks/models")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao testar os modelos: {str(e)}")

    return results
