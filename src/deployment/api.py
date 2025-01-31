from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Bem-vindo à API de Reclamações da NHTSA!"}
