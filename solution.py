import pandas as pd
import numpy as np

from scipy.stats import norm


chat_id = 263008738 # Ваш chat ID, не меняйте название переменной



def solution(p: float, x: np.array) -> tuple:
    # Измените код этой функции
    # Это будет вашим решением
    # Не меняйте название функции и её аргументы
    alpha = 1 - p
    loc = x.mean() + 1/2
    scale = np.sqrt(1) / np.sqrt(len(x))
    left = loc - scale * norm.ppf(1 - alpha / 2)
    right = loc - scale * norm.ppf(alpha / 2)
    l_a = (2*left)/(77**2)
    r_a = (2*right)/(77**2)
    return l_a, r_a
