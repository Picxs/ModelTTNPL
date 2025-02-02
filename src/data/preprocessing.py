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
from sklearn.metrics import roc_auc_score

stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Função para limpar e pré-processar o texto.
    
    Parâmetros:
    text: string contendo o texto que o usuário deseja processar.

    Retorna:
    Uma string do conjunto dos tokens criados, contendo texto processado e limpo.
    """
    lemmatizer = WordNetLemmatizer()
    
    # Aqui é utilizado o tokenize para transformar o texto que foi convertido para minúsculo em tokens, isso para melhor eficiência na hora de remover as stopwords e lematizar.
    tokens = word_tokenize(text.lower())
    #Aqui é utilizado o lemmatizer e o stopwords para limpar o texto.
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
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
    

    # Aqui todas as colunas foram nomeadas seguindo o CMPL.txt, retirado da aba de Complaints no site da NHTSA.
    df.columns = [
        "CMPLID", "ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "CRASH", "FAILDATE",
        "FIRE", "INJURED", "DEATHS", "COMPDESC", "CITY", "STATE", "VIN", "DATEA", "LDATE", "MILES",
        "OCCURENCES", "CDESCR", "CMPL_TYPE", "POLICE_RPT_YN", "PURCH_DT", "ORIG_OWNER_YN", "ANTI_BRAKES_YN",
        "CRUISE_CONT_YN", "NUM_CYLS", "DRIVE_TRAIN", "FUEL_SYS", "FUEL_TYPE", "TRANS_TYPE", "VEH_SPEED",
        "DOT", "TIRE_SIZE", "LOC_OF_TIRE", "TIRE_FAIL_TYPE", "ORIG_EQUIP_YN", "MANUF_DT", "SEAT_TYPE",
        "RESTRAINT_TYPE", "DEALER_NAME", "DEALER_TEL", "DEALER_CITY", "DEALER_STATE", "DEALER_ZIP", 
        "PROD_TYPE", "REPAIRED_YN", "MEDICAL_ATTN", "VEHICLES_TOWED_YN"
    ]
    
    # Aqui é aplicado o clean_text nas colunas de texto para fazer as operações que foram descritas.
    df["CDESCR"] = df["CDESCR"].astype(str).apply(clean_text)
    df["COMPDESC"] = df["COMPDESC"].astype(str).apply(clean_text)
    
    # Aqui as colunas que foram julgadas desnecessárias para o objetivo dos modelos são dropadas.
    df = df.drop(columns=[ 
        "VIN", "ODINO", "MFR_NAME", "MAKETXT", "CITY", "STATE", "DEALER_NAME", 
        "DEALER_TEL", "DEALER_CITY", "DEALER_STATE", "DEALER_ZIP", "DOT", "TIRE_SIZE", 
        "LOC_OF_TIRE", "SEAT_TYPE", "RESTRAINT_TYPE", "CMPLID", "FAILDATE",
        "DATEA", "LDATE", "MILES", "PURCH_DT", "NUM_CYLS", "VEH_SPEED", "ORIG_EQUIP_YN",
        "MANUF_DT", "REPAIRED_YN"
    ])
    
    
    # Aqui é separado as colunas em categóricas e númericas para fazer as transformações de cada tipo corretamente.
    categorical_columns = ["CMPL_TYPE", "DRIVE_TRAIN", "FUEL_SYS", "FUEL_TYPE", "TRANS_TYPE", "TIRE_FAIL_TYPE", "PROD_TYPE"]

    # Aqui foi feita a conversão da coluna para númerico, pois ela é registrada como char no DataFrame original, é preciso dela como numérico para entrar na normalização.
    df["YEARTXT"] = pd.to_numeric(df["YEARTXT"], errors='coerce')

    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    print("As colunas numéricas são: " + ", ".join(numeric_columns))


    if "INJURED" not in numeric_columns:
        print("Coluna 'INJURED' não encontrada nas colunas numéricas. Verifique o tipo da coluna.")
    if "DEATHS" not in numeric_columns:
        print("Coluna 'DEATHS' não encontrada nas colunas numéricas. Verifique o tipo da coluna.")

    # Aqui é onde as colunas que constam com 'Y' para true e 'N' para false são convertidas para binário, sendo agora 1 para 'Y' e 0 para 'N'.
    binary_columns = ["CRASH", "FIRE", "POLICE_RPT_YN", "ORIG_OWNER_YN", "ANTI_BRAKES_YN", 
                      "CRUISE_CONT_YN", "MEDICAL_ATTN", "VEHICLES_TOWED_YN"]
    for col in binary_columns:
        df[col] = df[col].map({'Y': 1, 'N': 0})

 
    
    # Aqui é feito a vetorização das colunas de texto que já estão processadas, gerando 50 novas features com base na importância das palavras. 
    text_columns = ['CDESCR', 'COMPDESC']
    vectorizer = TfidfVectorizer(max_features=50)
    # Aqui as duas colunas de texto são concatenadas para que seja feito apenas uma vetorização.
    combined_text = df[text_columns].apply(lambda x: ' '.join(x), axis=1)
    # Aqui é aplicado o TF-IDF, efetivamente vetorizando a coluna de texto
    tfidf_matrix = vectorizer.fit_transform(combined_text)
    # Aqui a matriz resultante da vetorização é convertida array e então para DataFrame, para poder ser concatenado novamente com o DataFrame principal.
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    
    # Aqui é dropado as colunas de texto que não são mais necessárias.
    df = df.drop(columns=text_columns)

    # Aqui é criado um pré-processador de dados que utiliza o ColumnTransformer, que é uma maneira de aplicar tranformações específicas a determinadas colunas do DataFrame.
    # Neste caso, será utilizado o StandardScaler, que normaliza os dados númericos 
    # e o OneHotEncoder, que é usado para converter variáveis categóricas em colunas binárias.
    preprocessor = ColumnTransformer([ 
        ('num', StandardScaler(), numeric_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns)
    ])
    # Aqui é aplicado o pré-processador.
    df_transformed = preprocessor.fit_transform(df)
    # Aqui é criado um novo DataFrame usando a matriz tranformada e definindo os nomes das colunas apropriadamente.
    df_transformed = pd.DataFrame(df_transformed, columns= 
                                  numeric_columns + 
                                  list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_columns)))
    
    # Aqui é feito a concatenação do TF-IDF com o restante dos dados transformados
    df_final = pd.concat([df_transformed, tfidf_df, df[binary_columns]], axis=1)

    # Aqui o DataFrame é salvo em .csv
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



def create_processed_file(input_file="data/raw/COMPLAINTS_RECEIVED_2015-2019.txt", output_dir="data/processed"):
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
    Função para preparar os dados e definir os modelos, 
    realizando a divisão de treino e teste.

    Parâmetros:
    df: DataFrame contendo os dados.

    Retorna:
    models: Dicionário com os modelos prontos para treino.
    X_train, X_test, y_train, y_test: Conjuntos de dados divididos para treino e teste.
    """

    # Aqui é feita a definição do target (y) e das features (X).
    y = df["MEDICAL_ATTN"]
    X = df.drop(columns=["MEDICAL_ATTN"])

    # Aqui fazemos aransformação para binário (1 = positivo, 0 = negativo) do target, pois é uma classificação binária.
    y = (y > 0).astype(int)

    # Aqui é feita a divisão de treino e teste, separando 20% dos dados para o teste.
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Aqui temos os 3 modelos que serão utilizados. Todos os parâmetros dos modelos foram pensados para tentar evitar o overfit.
    models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,  # Este parâmetros é muito importante pois evita árvores muito profundas, reduzindo overfitting.
        min_samples_split=20, 
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    
    "Logistic Regression": LogisticRegression(
        random_state=42,
        max_iter=5000,
        C=0.5,  # Aqui setamos uma regularização mais forte para evitar overfitting
        solver='saga',
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=15,  # Aqui mantemos a profundidade da árvore igual ao Random Forest, pelo mesmo motivo.
        subsample=0.7, 
        colsample_bytree=0.7,
        scale_pos_weight=1.0,  
        eval_metric='logloss',  # Aqui setamos a métrica comummente utilizada para classificação binária.
        tree_method='hist',  # Usa histograma para acelerar o treinamento
        random_state=42,
        n_jobs=-1
    )
}

    return models, X_train, X_test, y_train, y_test


def removing_nan(df):
    """
    Remove colunas do DataFrame que contêm '_nan' no nome.

    Parâmetros:
    df: O DataFrame original.

    Retorna:
    df: O DataFrame sem as colunas dos valores NaN
    """
    # Identificar colunas que contêm '_nan' no nome
    colunas_com_nan = [col for col in df.columns if '_nan' in col]
    
    # Dropar as colunas diretamente no DataFrame original
    df.drop(columns=colunas_com_nan, inplace=True)

    return df