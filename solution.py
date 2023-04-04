import pandas as pd
import numpy as np

from scipy.stats import norm


chat_id = 263008738 # Ваш chat ID, не меняйте название переменной

def solution(p: float, x: np.array) -> tuple:
    n_samples = 1000 # количество выборок для бутстрэпа
    n = len(x)
    a_values = np.zeros(n_samples)
    for i in range(n_samples):
        X_bootstrap = np.random.choice(X, size=n, replace=True)
        mean_X = np.mean(X_bootstrap)
        std_X = np.std(X_bootstrap, ddof=1)
        a_bootstrap = 2 * mean_X / (77 ** 2) # коэффициент ускорения на этой выборке
        a_values[i] = a_bootstrap
    alpha = 1 - p
    z_value = norm.ppf(1 - alpha / 2) # Z-значение для заданного уровня доверия
    mean_a = np.mean(a_values)
    std_a = np.std(a_values, ddof=1)
    lower_bound = mean_a - z_value * std_a / np.sqrt(n_samples)
    upper_bound = mean_a + z_value * std_a / np.sqrt(n_samples)
    return (lower_bound, upper_bound)
