import numpy as np
from copy import deepcopy

class Function:
    """目的関数のためのクラス
    
    Attributes:
        h (np.ndarray): 勾配を計算するための微小変化値。shapeは(dim, )
    """
    def __init__(self, h:np.ndarray = None)->None:
        self.h = h
    def __call__(self, x:np.ndarray)->float:
        """目的関数値を出力

        forwardで計算された目的関数値を修正する必要がある場合は、ここに記入する。
        Args:
            x (np.ndarray): 入力。shapeは(dim, )
        Returns:
            float: 目的関数値
        """
        return self.forward(x)
    def forward(self, x:np.ndarray)->float:
        """目的関数値を出力

        Args:
            x (np.ndarray): 入力。shapeは(dim, )
        Raises:
            NotImplementedError: 要継承
        Returns:
            float: 目的関数値
        """
        raise NotImplementedError
    
    def grad(self, x:np.ndarray)->np.ndarray:
        """勾配値を計算

        継承されない場合、差分法で計算
        Args:
            x (np.ndarray): 入力。shapeは(dim, )
        Returns:
            np.ndarray: 勾配。shapeは(dim, )
        """
        assert self.h is not None, "Overwrite grad or define h"
        assert len(self.h) == len(x), "Dimension of h is different from one of x"

        y = self(x)
        g = []
        for i in range(len(x)):
            x_new = deepcopy(x)
            x_new[i] += self.h[i]
            y_new = self(x_new)
            g.append((y_new-y)/self.h[i])
        
        return np.array(g)

class PenaltyFunction(Function):
    """ペナルティ関数用のクラス

    Attributes:
        h (np.ndarray): 勾配を計算するための微小変化値。shapeは(dim, )
    """
    def __call__(self, x:np.ndarray)->None:
        p = self.forward(x)
        return np.max(p, 0.)

class Optimizer:
    """最適化手法の抽象クラス

    Attributes:
        objective (list[Function]): 目的関数のリスト。
        objective_w (np.ndarray): 各目的関数の重み。
        penalty (list[PenaltyFunction]): ペナルティ関数のリスト。Noneの場合制約なし最適化。
        penalty_w (np.ndarray): 各ペナルティ関数の重み。
    Note:
        * 重み付き加算による多目的最適化に対応
    """
    def __init__(self,
                objective:list[Function],
                objective_w:np.ndarray = np.ones(1),
                penalty:list[PenaltyFunction] = None,
                penalty_w:np.ndarray = None,)->None:
        assert len(objective) == objective_w
        if penalty is not None:
            assert len(penalty) == len(penalty_w)
        self.objective = objective
        self.objective_w = objective_w
        self.penalty = penalty
        self.penalty_w = penalty_w
    
    def forward(self, x:np.ndarray)->tuple[float, np.ndarray, np.ndarray]:
        """評価値を出力

        Args:
            x (np.ndarray): 入力
        Returns:
            tuple[float, np.ndarray, np.ndarray]: 評価値、目的関数値のリスト、ペナルティ関数値のリスト
        """
        objectives = np.array([obj(x) for obj in self.objective])
        if self.penalty is not None:
            penalties = np.array([p(x) for p in self.penalty])
            return np.sum(self.objective_w*objectives + self.penalty_w*penalties), objectives, penalties
        else:
            return np.sum(self.objective_w*objectives), objectives, None
    
    def grad(self, x:np.ndarray)->float:
        """評価値に対する勾配を計算

        Args:
            x (np.ndarray): 入力
        Returns:
            float: 勾配。
        """
        g = np.zeros(x.shape)
        for obj, w in zip(self.objective, self.objective_w):
            g += w*obj.grad(x)
        
        if self.penalty is not None:
            for p, w in zip(self.penalty, self.penalty_w):
                g += w*p.grad(x)
        
        return g
    
    def step(self, x:np.ndarray)->np.ndarray:
        """候補解xから解を更新

        Args:
            x (np.ndarray): 候補解
        Raises:
            NotImplementedError: 要継承
        Returns:
            np.ndarray: 新しい候補解
        """
        raise NotImplementedError