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
        """解を更新

        Args:
            x (np.ndarray): 入力
            d (np.ndarray, optional): 進行方向。1反復計算目はNone。詳しくは書籍参照。
        Returns:
            tuple[np.ndarray]: 新しい候補解と新しい進行方向。
        """
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
        """解を更新

        Args:
            x (np.ndarray): 候補解
            B (np.ndarray, optional): BFGS行列。詳しくは書籍参照。
        Returns:
            tuple[np.ndarray]: 新しい候補解と新しいBFGS行列。
        """
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
    """記憶制限付きBFGS

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
        """解を更新

        Args:
            x (np.ndarray): 候補解
            s (np.ndarray, optional): 書籍参照。1反復計算目はNone。
            y (np.ndarray, optional): 書籍参照。1反復計算目はNone。
            c (np.ndarray, optional): 書籍参照。1反復計算目はNone。

        Returns:
            tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray]]: 更新結果
        """
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


class Momentum(SteepestGradientDescent):
    """Momentum法

    Attributes:
        objective (list[Function]): 目的関数のリスト。
        objective_w (np.ndarray): 各目的関数の重み。
        penalty (list[PenaltyFunction]): ペナルティ関数のリスト。Noneの場合制約なし最適化。
        penalty_w (np.ndarray): 各ペナルティ関数の重み。
        eta (float): 速度緩和係数。
        alpha (float): 緩和係数。
    """
    def __init__(self,
                objective:list[Function],
                objective_w:np.ndarray = np.ones(1),
                penalty:list[PenaltyFunction] = None,
                penalty_w:np.ndarray = None,
                eta:float = 0.9,
                alpha:float = 1.)->None:
        super().__init__(objective, objective_w, penalty, penalty_w)
        self.alpha = alpha
        self.eta = eta
    
    def step(self, x:np.ndarray, v:np.ndarray = None)->tuple[np.ndarray]:
        """解の更新

        Args:
            x (np.ndarray): 候補解
            v (np.ndarray, optional): 速度。1反復計算目はゼロ。
        Returns:
            tuple[np.ndarray]: 新しい候補解と新しい速度
        """
        if v is None: v = np.zeros(len(x))
        g = self.grad(x)
        v_new = self.eta*v - self.alpha*g

        x_new = x + v_new
        return x_new, v