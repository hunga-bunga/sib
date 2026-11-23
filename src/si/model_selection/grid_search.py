import itertools
import numpy as np
import copy
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from _search_utils import kfold_indices
from _search_utils import cross_validate_configuration

#função grid_search
def grid_search_cv(model, dataset, hyperparameter_grid, scoring=None, cv=5, seed=None):
    """
    Realiza uma pesquisa em grelha com validação cruzada para otimização de hiperparâmetros[cite: 1].
    """
    X, y = dataset
    n_samples = len(X)
    
    # Criar os k folds uma vez e reutilizar
    folds = kfold_indices(n_samples, cv=cv, seed=seed)
    
    keys = hyperparameter_grid.keys()
    values = hyperparameter_grid.values()
    
    # Gerar todas as combinações possíveis de hiperparâmetros
    all_combinations = list(itertools.product(*values))
    
    tested_hyperparameters = []
    mean_scores = []
    
    best_score = -float('inf')
    best_hyperparameters = None
    
    for combination in all_combinations:
        param_dict = dict(zip(keys, combination))
        tested_hyperparameters.append(param_dict)
        
        # Avaliar a configuração atual com validação cruzada
        mean_score = cross_validate_configuration(
            model=model,
            dataset=dataset,
            hyperparameters=param_dict,
            scoring=scoring,
            cv=folds,
            seed=seed
        )
        
        mean_scores.append(mean_score)
        
        # Atualizar a melhor configuração se aplicável
        if mean_score > best_score:
            best_score = mean_score
            best_hyperparameters = param_dict
            
    return {
        "hyperparameters": tested_hyperparameters,
        "scores": mean_scores,
        "best_hyperparameters": best_hyperparameters,
        "best_score": best_score
    }


#TESTE

if __name__ == "__main__":
    # Criar um dataset sintético de classificação
    X_synth, y_synth = make_classification(n_samples=150, n_features=10, random_state=42)
    dataset = (X_synth, y_synth)
    
    # Definir a grelha de hiperparâmetros a testar
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [3, 5, None]
    }
    
    print("A executar o Grid Search")
    
    # Executar a função com a classe do modelo (RandomForestClassifier)
    resultados = grid_search_cv(
        model=RandomForestClassifier,
        dataset=dataset,
        hyperparameter_grid=param_grid,
        cv=5,
        seed=42
    )
    
    print("\n--- RESULTADOS DO TESTE ---")
    print(f"Melhores Hiperparâmetros: {resultados['best_hyperparameters']}")
    print(f"Melhor Pontuação Média: {resultados['best_score']:.4f}")
    print(f"\nTotal de configurações testadas: {len(resultados['hyperparameters'])}")