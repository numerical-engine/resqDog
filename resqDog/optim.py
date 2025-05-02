from resqDog.core import Optimizer
import numpy as np
from resqDog.core import Function, PenaltyFunction
import sys
from copy import deepcopy

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


class ConjugateGradient(SteepestGradientDescent):
    """共役勾配法

    Attributes:
        objective (list[Function]): 目的関数のリスト。
        objective_w (np.ndarray): 各目的関数の重み。
        penalty (list[PenaltyFunction]): ペナルティ関数のリスト。Noneの場合制約なし最適化。
        penalty_w (np.ndarray): 各ペナルティ関数の重み。
        alpha (float): 緩和係数。
    """
    def step(self, x:np.ndarray, d:np.ndarray = None)->tuple[np.ndarray]:
        g = self.grad(x)
        if d is None:
            d = -g
        x_new = x + self.alpha*d
        g_new = self.grad(x_new)
        beta = (np.linalg.norm(g_new)**2)/(np.linalg.norm(g)**2)
        d_new = -g_new + beta*d

        return x_new, d_new


class BFGS(SteepestGradientDescent):
    """BFGS公式に基づく準ニュートン法

    Attributes:
        objective (list[Function]): 目的関数のリスト。
        objective_w (np.ndarray): 各目的関数の重み。
        penalty (list[PenaltyFunction]): ペナルティ関数のリスト。Noneの場合制約なし最適化。
        penalty_w (np.ndarray): 各ペナルティ関数の重み。
        alpha (float): 緩和係数。
    """
    def step(self, x:np.ndarray, B:np.ndarray = None)->tuple[np.ndarray]:
        if B is None: B = np.eye(len(x))

        B_inv = np.linalg.pinv(B) #逆行列を計算。数値誤差対策でpinvを使用。
        g = self.grad(x)
        d = -B_inv@g

        x_new = x + self.alpha*d
        g_new = self.grad(x_new)
        y = (g_new - g).reshape((-1,1))
        s = (x_new - x).reshape((-1,1))

        B_new = B - (B@(s@s.T)@B)/(s.T@(B@s)) + (y@y.T)/(s.T@y)

        return x_new, B_new


class LBFGS(BFGS):
    """最急降下法

    Attributes:
        objective (list[Function]): 目的関数のリスト。
        objective_w (np.ndarray): 各目的関数の重み。
        penalty (list[PenaltyFunction]): ペナルティ関数のリスト。Noneの場合制約なし最適化。
        penalty_w (np.ndarray): 各ペナルティ関数の重み。
        alpha (float): 緩和係数。
        m (int): 記憶数。
    """
    def __init__(self,
                objective:list[Function],
                objective_w:np.ndarray = np.ones(1),
                penalty:list[PenaltyFunction] = None,
                penalty_w:np.ndarray = None,
                alpha:float = 1.,
                m:int = 1)->None:
        super().__init__(objective, objective_w, penalty, penalty_w)
        self.alpha = alpha
        self.m = m
    def step(self,
            x:np.ndarray,
            s:np.ndarray = None,
            y:np.ndarray = None,
            c:np.ndarray = None,)->tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        g = self.grad(x)
        if s is None:
            x_new = x - self.alpha*g
            s = [x_new - x]
            y = [self.grad(x_new) - g]
            c = [1./np.sum(s[-1]*y[-1])]
            return x_new, s, y, c
        else:
            sl = s[-1].reshape((-1,1)); yl = y[-1].reshape((-1,1))
            H = (sl.T@yl)/(yl.T@yl)*np.eye(len(x))
            q = deepcopy(g)
            tau = []
            for si, yi, ci in zip(reversed(s), reversed(y), reversed(c)):
                tau.append(ci*np.sum(si*q))
                q -= tau[-1]*yi
            d = H@q
            for si, yi, ci, ti in zip(s, y, c, tau):
                d = d - (ci*np.sum(yi*d))*si + ti*si
            x_new = x - self.alpha*d
            y.append(self.grad(x_new)-g)
            s.append(x_new - x)
            c.append(1./np.sum(s[-1]*y[-1]))

            if len(y) > self.m:
                y = y[1:]; s = s[1:]; c = c[1:]
            return x_new, s, y, c