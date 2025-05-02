from resqDog.core import Optimizer
import numpy as np
from resqDog.core import Function, PenaltyFunction
import sys

class SteepestGradientDescent(Optimizer):
    """最急降下法

    Attributes:
        objective (list[Function]): 目的関数のリスト。
        objective_w (np.ndarray): 各目的関数の重み。
        penalty (list[PenaltyFunction]): ペナルティ関数のリスト。Noneの場合制約なし最適化。
        penalty_w (np.ndarray): 各ペナルティ関数の重み。
        alpha (float): 緩和係数。
    """
    def __init__(self,
                objective:list[Function],
                objective_w:np.ndarray = np.ones(1),
                penalty:list[PenaltyFunction] = None,
                penalty_w:np.ndarray = None,
                alpha:float = 1.)->None:
        super().__init__(objective, objective_w, penalty, penalty_w)
        self.alpha = alpha

    def step(self, x:np.ndarray)->np.ndarray:
        g = self.grad(x)
        return x - self.alpha*g