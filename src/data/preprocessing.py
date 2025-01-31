import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Função para limpar e pré-processar o texto.
    
    Parâmetros:
    text: string contendo o texto que o usuário deseja processar.
    """
    
    lemmatizer = WordNetLemmatizer()

    tokens = word_tokenize(text.lower())  # Tokenizar e converter para minúsculo
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]  # Remover stopwords e lematizar
    return ' '.join(tokens)

def process_data(input_file):
    """
    Aplica o pré-processamento e limpeza nos dados para o treinamento ser feito de maneira eficiente e eficaz,
    salva o arquivo processado com as modificações.
    
    Parâmetros:
    input_file: string contendo o path para o arquivo que será feito o pré-processamento e limpeza.
    
    """

    if not os.path.exists(input_file):
        print(f"Erro: O arquivo '{input_file}' não foi encontrado.")
        return
    
    try:
        df = pd.read_csv(input_file)
        print("Arquivo carregado com sucesso! Primeiras linhas:")
        print(df.head())
    except Exception as e:
        print(f"Erro ao ler o arquivo: {e}")
        return
    
    df.columns = [
        "CMPLID", "ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "CRASH", "FAILDATE",
        "FIRE", "INJURED", "DEATHS", "COMPDESC", "CITY", "STATE", "VIN", "DATEA", "LDATE", "MILES",
        "OCCURENCES", "CDESCR", "CMPL_TYPE", "POLICE_RPT_YN", "PURCH_DT", "ORIG_OWNER_YN", "ANTI_BRAKES_YN",
        "CRUISE_CONT_YN", "NUM_CYLS", "DRIVE_TRAIN", "FUEL_SYS", "FUEL_TYPE", "TRANS_TYPE", "VEH_SPEED",
        "DOT", "TIRE_SIZE", "LOC_OF_TIRE", "TIRE_FAIL_TYPE", "ORIG_EQUIP_YN", "MANUF_DT", "SEAT_TYPE",
        "RESTRAINT_TYPE", "DEALER_NAME", "DEALER_TEL", "DEALER_CITY", "DEALER_STATE", "DEALER_ZIP", 
        "PROD_TYPE", "REPAIRED_YN", "MEDICAL_ATTN", "VEHICLES_TOWED_YN"
    ]
    
    # Limpeza das colunas de texto
    df["CDESCR"] = df["CDESCR"].astype(str).apply(clean_text)
    df["COMPDESC"] = df["COMPDESC"].astype(str).apply(clean_text)
    
    # Excluindo as colunas não necessárias
    df = df.drop(columns=[ 
        "VIN", "ODINO", "MFR_NAME", "MAKETXT", "CITY", "STATE", "DEALER_NAME", 
        "DEALER_TEL", "DEALER_CITY", "DEALER_STATE", "DEALER_ZIP", "DOT", "TIRE_SIZE", 
        "LOC_OF_TIRE", "SEAT_TYPE", "RESTRAINT_TYPE", "CMPLID", "FAILDATE",
        "DATEA", "LDATE", "MILES", "PURCH_DT", "NUM_CYLS", "VEH_SPEED", "ORIG_EQUIP_YN",
        "MANUF_DT", "REPAIRED_YN"
    ])
    
    # Transformação de dados binários
    binary_columns = ["CRASH", "FIRE", "POLICE_RPT_YN", "ORIG_OWNER_YN", "ANTI_BRAKES_YN", 
                      "CRUISE_CONT_YN", "MEDICAL_ATTN", "VEHICLES_TOWED_YN"]
    for col in binary_columns:
        df[col] = df[col].map({'Y': 1, 'N': 0})
    
    # Colunas categóricas e numéricas
    categorical_columns = ["CMPL_TYPE", "DRIVE_TRAIN", "FUEL_SYS", "FUEL_TYPE", "TRANS_TYPE", "TIRE_FAIL_TYPE", "PROD_TYPE"]
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    # Vetorização das colunas de texto (CDESCR, COMPDESC)
    text_columns = ['CDESCR', 'COMPDESC']
    vectorizer = TfidfVectorizer(max_features=50)
    
    # Concatenando as duas colunas de texto para aplicação do TF-IDF
    combined_text = df[text_columns].apply(lambda x: ' '.join(x), axis=1)
    
    # Aplicando o TF-IDF
    tfidf_matrix = vectorizer.fit_transform(combined_text)
    
    # Convertendo a matriz TF-IDF em um DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    
    # Removendo as colunas de texto antes de processar os dados restantes
    df = df.drop(columns=text_columns)

    # Preprocessamento das colunas numéricas e categóricas
    preprocessor = ColumnTransformer([ 
        ('num', StandardScaler(), numeric_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns)
    ])
    
    # Aplicando a transformação apenas nas colunas numéricas e categóricas
    df_transformed = preprocessor.fit_transform(df)
    
    # Criando um DataFrame com as colunas processadas
    df_transformed = pd.DataFrame(df_transformed, columns= 
                                  numeric_columns + 
                                  list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_columns)))
    
    # Concatenando o TF-IDF com o restante dos dados transformados
    df_final = pd.concat([df_transformed, tfidf_df], axis=1)

    # Salvando o DataFrame processado
    output_file = input_file.replace("raw", "processed")
    try:
        df_final.to_csv(output_file, index=False)
        print(f"Arquivo processado salvo em: {output_file}")
    except Exception as e:
        print(f"Erro ao salvar o arquivo: {e}")
        return

    if os.path.exists(output_file):
        print(f"Arquivo CSV processado com sucesso: {output_file}")
    else:
        print(f"Erro: O arquivo CSV não foi gerado.")


def create_processed_file(input_file="models/data/raw/COMPLAINTS_RECEIVED_2015-2019.txt", output_dir="models/data/processed"):
    """
    Cria um arquivo processado .csv a partir do arquivo de dados bruto baixado.
    
    Parâmetros:
    input_file: string contendo o path para o arquivo de dados bruto que vai ser processado e transformado em .csv.
    output_dir: string contendo o path para o diretório onde o arquivo processado vai ser criado e salvo.
    """

    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_file):
        print(f"Erro: O arquivo '{input_file}' não foi encontrado.")
        return
    
    try:
        df = pd.read_csv(input_file, sep="\t", header=None)
        print("Arquivo carregado com sucesso! Primeiras linhas:")
        print(df.head())
    except Exception as e:
        print(f"Erro ao ler o arquivo: {e}")
        return
    
    consolidated_file = os.path.join(output_dir, "complaints.csv")
    try:
        df.to_csv(consolidated_file, index=False)
        print(f"Arquivo carregado salvo em: {consolidated_file}")
    except Exception as e:
        print(f"Erro ao salvar o arquivo: {e}")
        return
    
    if os.path.exists(consolidated_file):
        print(f"Arquivo CSV gerado com sucesso: {consolidated_file}")
        process_data(consolidated_file)
    else:
        print(f"Erro: O arquivo CSV não foi gerado.")

def impute_missing_values(df):
    """
    Função que realiza imputação de valores ausentes nas colunas numéricas
    e categóricas, e salva as alterações no DataFrame.
    
    Parâmetros:
    df: DataFrame com dados a serem imputados.
    
    Retorna:
    df: DataFrame com valores ausentes imputados.
    """
    
    # Identificando colunas numéricas e categóricas
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Verifique se há colunas categóricas antes de tentar imputação
    if len(categorical_cols) > 0:
        print(f"Colunas categóricas encontradas: {categorical_cols}")
    else:
        print("Nenhuma coluna categórica encontrada.")

    # Se houver dados válidos nas colunas categóricas, prosseguimos com a imputação
    for col in categorical_cols:
        # Verifica se há valores nulos e imprime o número de valores ausentes
        print(f"Valores ausentes na coluna {col}: {df[col].isna().sum()}")

        # Garantir que as colunas categóricas estão no tipo 'category'
        df[col] = df[col].astype('category')

    # Imputação para colunas numéricas
    numerical_imputer = SimpleImputer(strategy='mean')

    # Imputação para colunas categóricas
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    # Preenchendo os valores ausentes nas colunas numéricas
    df[numerical_cols] = numerical_imputer.fit_transform(df[numerical_cols])

    # Verifique novamente se há dados válidos nas colunas categóricas
    if len(categorical_cols) > 0:
        # Preenchendo os valores ausentes nas colunas categóricas
        df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

    return df

def prepare_models_and_split(df):
    """
    Função para preparar os dados e os modelos, realizar a divisão de treino e teste.

    Parâmetros:
    df: DataFrame contendo os dados.
    
    Retorna:
    models: Dicionário com os modelos treinados.
    X_train, X_test, y_train, y_test: Conjuntos de dados divididos para treino e teste.
    """
    # Definição das variáveis X e y
    y = df["MEDICAL_ATTN"]
    X = df.drop(columns=["MEDICAL_ATTN"])

    # Transformar para binário (positivo = 1, negativo ou zero = 0)
    y = (y > 0).astype(int)

    # Divisão dos dados com estratificação
    test_size = 0.2  # 20% dos dados para teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Definição dos modelos
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),  # Substituindo SVM por Regressão Logística
        "XGBoost": XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42)
    }
    
    return models, X_train, X_test, y_train, y_test
