import pandas as pd

# Exemplo de estrutura simples da classe Dataset
class Dataset:
    def __init__(self, X=None, y=None, feature_names=None):
        self.X = X                  # Matriz de características (features)
        self.y = y                  # Vetor de rótulos (labels/y)
        self.feature_names = feature_names

def read_csv(filename, sep=',', features=False, label=False):
    """
    Lê um ficheiro CSV e devolve um objeto Dataset.
    
    :param filename: Nome/caminho do ficheiro
    :param sep: Separador de valores
    :param features: Boolean. Indica se o ficheiro tem cabeçalho com nomes das características
    :param label: Boolean. Indica se a última coluna é o rótulo (y)
    :return: Objeto Dataset
    """
    # Define se lê o cabeçalho ou não com base no parâmetro 'features'
    header_option = 0 if features else None
    
    # Lê o ficheiro com o pandas
    df = pd.read_csv(filename, sep=sep, header=header_option)
    
    # Guarda os nomes das características se existirem no ficheiro
    feature_names = list(df.columns) if features else None
    
    # Processa o rótulo (y) e as características (X)
    if label:
        # Se 'label' for verdadeiro, a última coluna é o 'y'
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        if feature_names:
            feature_names = feature_names[:-1]
    else:
        X = df.values
        y = None
        
    return Dataset(X=X, y=y, feature_names=feature_names)


