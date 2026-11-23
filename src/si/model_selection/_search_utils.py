import numpy as np

def kfold_indices(n_samples, cv=5, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    folds = np.array_split(indices, cv)
    
    splits = []
    for i in range(cv):
        val_indices = folds[i]
        train_indices = np.concatenate([folds[j] for j in range(cv) if j != i])
        splits.append((train_indices, val_indices))
        
    return splits

import numpy as np
import copy

def cross_validate_configuration(model, dataset, hyperparameters, scoring=None, cv=5, seed=None):
    """
    Avalia uma única configuração de hiperparâmetros utilizando validação cruzada.
    
    Parâmetros:
    - model: o modelo a afinar ou função construtora.
    - dataset: o tuplo (X, y) com os dados de treino.
    - hyperparameters: dicionário com a configuração específica a testar {nome: valor}.
    - scoring: função de scoring opcional.
    - cv: número de folds ou lista de índices de folds pré-calculada.
    - seed: semente para a criação dos folds (caso cv seja um inteiro).
    
    Retorna:
    - A pontuação média (mean score) obtida nos folds.
    """
    X, y = dataset
    n_samples = len(X)
    
    # Se 'cv' for um inteiro, geramos os folds; se já for uma lista de folds, reutilizamos
    if isinstance(cv, int):
        folds = kfold_indices(n_samples, cv=cv, seed=seed)
    else:
        folds = cv
        
    fold_scores = []
    
    for train_index, val_index in folds:
        if hasattr(X, "iloc"):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        else:
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
        
        # Instanciar ou clonar o modelo com os hiperparâmetros atuais
        if callable(model):
            current_model = model(**hyperparameters)
        else:
            current_model = copy.deepcopy(model)
            if hasattr(current_model, "set_params"):
                current_model.set_params(**hyperparameters)
            else:
                for k, v in hyperparameters.items():
                    setattr(current_model, k, v)
                    
        # Treinar o modelo
        current_model.fit(X_train, y_train)
        
        # Avaliar o modelo
        if scoring is not None:
            score = scoring(current_model, X_val, y_val)
        elif hasattr(current_model, "score"):
            score = current_model.score(X_val, y_val)
        else:
            raise ValueError("O modelo não tem um método .score() e nenhuma função 'scoring' foi fornecida.")
            
        fold_scores.append(score)
        
    # Retorna a pontuação média da validação cruzada para esta configuração
    return np.mean(fold_scores)


