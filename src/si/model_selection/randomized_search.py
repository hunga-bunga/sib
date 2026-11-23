from _search_utils import kfold_indices 
from _search_utils import cross_validate_configuration
import numpy as np

def log_uniform(low, high):
    """Retorna uma função que amostra 10 ** uniform(log10(low), log10(high))[cite: 2]."""
    def sampler(rng):
        log_low = np.log10(low)
        log_high = np.log10(high)
        return 10.0 ** rng.uniform(log_low, log_high)
    return sampler

def sample_configuration(hyperparameter_distributions, rng):
    """Amostra uma única configuração de hiperparâmetros com base nas distribuições."""
    param_dict = {}
    for name, dist in hyperparameter_distributions.items():
        if isinstance(dist, (list, tuple, np.ndarray)):
            # Lista -> escolha uniforme[cite: 2]
            param_dict[name] = rng.choice(dist)
        elif callable(dist):
            # Função (ex: log_uniform) -> chamá-la com o gerador[cite: 2]
            param_dict[name] = dist(rng)
        else:
            param_dict[name] = dist
    return param_dict

def randomized_search_cv(model, dataset, hyperparameter_distributions, scoring=None, cv=5, n_iter=10, seed=None):
    """
    Realiza uma pesquisa aleatória com validação cruzada para otimização de hiperparâmetros[cite: 2].
    """
    X, y = dataset
    n_samples = len(X)
    
    # 1. Criar os k folds uma vez e um gerador aleatório com a seed[cite: 2]
    folds = kfold_indices(n_samples, cv=cv, seed=seed)
    rng = np.random.default_rng(seed)
    
    tested_hyperparameters = []
    mean_scores = []
    
    best_score = -float('inf')
    best_hyperparameters = None
    
    # 2. Repetir n_iter vezes[cite: 2]
    for _ in range(n_iter):
        # Amostrar a configuração usando a função sample_configuration
        param_dict = sample_configuration(hyperparameter_distributions, rng)
        tested_hyperparameters.append(param_dict)
        
        # 3. Avaliar a configuração gerada usando cross_validate_configuration
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
            
    # 4. Retornar o dicionário correspondente[cite: 2]
    return {
        "hyperparameters": tested_hyperparameters,
        "scores": mean_scores,
        "best_hyperparameters": best_hyperparameters,
        "best_score": best_score
    }