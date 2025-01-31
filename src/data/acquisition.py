import os
import urllib.request
import zipfile
import argparse

def fetch_data(download_url, output_dir="models/data/raw"):
    """
    Função que baixa os dados de reclamações a partir de uma URL fornecida usando urllib e descompacta o arquivo ZIP.
    
    Parâmetros:
    download_url: string contendo a url para download
    output_dir: diretório onde será salvo os dados descompactados.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Nome do arquivo ZIP e caminho de saída
    file_name = download_url.split("/")[-1]
    output_path = os.path.join(output_dir, file_name)
    
    # Verifica se o diretório já contém arquivos extraídos
    extracted_files = [f for f in os.listdir(output_dir) if f != file_name]
    
    if extracted_files:
        print(f"Os arquivos já foram extraídos em {output_dir}. Nenhuma ação necessária.")
        return  # Sai da função
    
    try:
        # Fazendo o download do arquivo
        urllib.request.urlretrieve(download_url, output_path)
        print(f"Arquivo ZIP salvo em: {output_path}")
        
        # Descompactando o arquivo ZIP
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
            print(f"Arquivos descompactados em: {output_dir}")
        
        # Removendo o arquivo ZIP após descompactar
        os.remove(output_path)
        print(f"Arquivo ZIP {file_name} removido.")
    
    except Exception as e:
        print(f"Erro ao baixar ou descompactar os dados: {e}")