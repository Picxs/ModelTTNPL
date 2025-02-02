import os
import urllib.request
import zipfile
import argparse

def fetch_data(download_url, output_dir="data/raw"):
    """
    Função que baixa os dados de reclamações a partir de uma URL fornecida usando urllib e descompacta o arquivo ZIP.
    
    Parâmetros:
    download_url: string contendo a url para download
    output_dir: diretório onde será salvo os dados descompactados.
    """
    
    os.makedirs(output_dir, exist_ok=True)

    file_name = download_url.split("/")[-1]
    output_path = os.path.join(output_dir, file_name)
    
    # Nesta linha o os. checa se o arquivo já foi extraido, para que o processo não fique redundante e nem gere cópias desnecessárias
    extracted_files = [f for f in os.listdir(output_dir) if f != file_name]
    
    if extracted_files:
        print(f"Os arquivos já foram extraídos em {output_dir}. Nenhuma ação necessária.")
        return
    
    try:
        # Aqui é utilizado o urllib.request para fazer o download do .zip do DataFrame usando a url  
        urllib.request.urlretrieve(download_url, output_path)
        print(f"Arquivo ZIP salvo em: {output_path}")
        
        # Aqui é utilizado o zipfile para descompactar o .zip do DataFrame
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
            print(f"Arquivos descompactados em: {output_dir}")
        
        # Aqui é removido o arquivo .zip que não é mais necessário.
        os.remove(output_path)
        print(f"Arquivo ZIP {file_name} removido.")
    
    except Exception as e:
        print(f"Erro ao baixar ou descompactar os dados: {e}")