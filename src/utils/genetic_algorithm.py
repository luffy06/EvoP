import numpy as np
from types import MethodType, FunctionType
import warnings
import sys
import multiprocessing
import logging
from abc import ABCMeta, abstractmethod
from  sko.base import SkoBase
from sko.GA import GeneticAlgorithmBase
from sko.operators import ranking, selection, crossover, mutation
from sko.tools import set_run_mode
from functools import lru_cache
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

def func_w_kwargs_transformer(func, n_processes):
    '''
    transform this kind of function:
    ```
    def demo_func(x):
        x1, x2, x3 = x
        return x1 ** 2 + x2 ** 2 + x3 ** 2
    ```
    into this kind of function:
    ```
    def demo_func(x):
        x1, x2, x3 = x[:,0], x[:,1], x[:,2]
        return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2
    ```
    getting vectorial performance if possible:
    ```
    def demo_func(x):
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2
    ```
    :param func:
    :return:
    '''

    # to support the former version
    if (func.__class__ is FunctionType) and (func.__code__.co_argcount > 1):
        warnings.warn('multi-input might be deprecated in the future, use fun(p) instead')

        def func_transformed(X, kwargs):
            return np.array([func(x, **kwargs) for x in X])

        return func_transformed

    # to support the former version
    if (func.__class__ is MethodType) and (func.__code__.co_argcount > 2):
        warnings.warn('multi-input might be deprecated in the future, use fun(p) instead')

        def func_transformed(X, kwargs):
            return np.array([func(x, **kwargs) for x in X])

        return func_transformed

    # to support the former version
    if getattr(func, 'is_vector', False):
        warnings.warn('''
        func.is_vector will be deprecated in the future, use set_run_mode(func, 'vectorization') instead
        ''')
        set_run_mode(func, 'vectorization')

    mode = getattr(func, 'mode', 'others')
    valid_mode = ('common', 'multithreading', 'multiprocessing', 'vectorization', 'cached', 'joblib', 'others')
    assert mode in valid_mode, 'valid mode should be in ' + str(valid_mode)
    logger.info(f"func mode: {mode}")
    if mode == 'vectorization':
        return func
    elif mode == 'cached':
        @lru_cache(maxsize=None)
        def func_cached(x, kwargs):
            return func(x, **kwargs)

        def func_warped(X, kwargs):
            return np.array([func_cached(x, kwargs) for x in X])

        return func_warped
    elif mode == 'multithreading':
        assert n_processes >= 0, 'n_processes should >= 0'
        from multiprocessing.dummy import Pool as ThreadPool
        if n_processes == 0:
            pool = ThreadPool()
        else:
            pool = ThreadPool(n_processes)

        def func_transformed(X, kwargs):
            return np.array(pool.map(func, X, **kwargs))

        return func_transformed
    elif mode == 'multiprocessing':
        assert n_processes >= 0, 'n_processes should >= 0'
        from multiprocessing import Pool
        if n_processes == 0:
            pool = Pool()
        else:
            pool = Pool(n_processes)
        def func_transformed(X, kwargs):
            return np.array(pool.map(func, X, **kwargs))

        return func_transformed
        
    elif mode == "joblib":
        from joblib import Parallel, delayed
        def func_transformed(X, kwargs):
            res = Parallel(n_jobs=-1, batch_size='auto')(
                delayed(func)(x, **kwargs) for x in X
            )
            return np.array(res)
        return func_transformed
        
    else:  # common
        def func_transformed(X, kwargs):
            return np.array([func(x, **kwargs) for x in X])

        return func_transformed


class GA(GeneticAlgorithmBase):
    """genetic algorithm

    Parameters
    ----------------
    func : function
        The func you want to do optimal
    n_dim : int
        number of variables of func
    lb : array_like
        The lower bound of every variables of func
    ub : array_like
        The upper bound of every variables of func
    constraint_eq : tuple
        equal constraint
    constraint_ueq : tuple
        unequal constraint
    precision : array_like
        The precision of every variables of func
    size_pop : int
        Size of population
    max_iter : int
        Max of iter
    prob_mut : float between 0 and 1
        Probability of mutation
    n_processes : int
        Number of processes, 0 means use all cpu
    Attributes
    ----------------------
    Lind : array_like
         The num of genes of every variable of func（segments）
    generation_best_X : array_like. Size is max_iter.
        Best X of every generation
    generation_best_ranking : array_like. Size if max_iter.
        Best ranking of every generation
    Examples
    -------------
    https://github.com/guofei9987/scikit-opt/blob/master/examples/demo_ga.py
    """

    def __init__(self, func, func_kwargs, n_dim, n_ones,
                 size_pop=50, max_iter=100,
                 prob_mut=0.001,
                 lb=-1, ub=1,
                 constraint_eq=tuple(), constraint_ueq=tuple(),
                 precision=1e-7, early_stop=None, n_processes=0):
        super().__init__(func, n_dim, size_pop, max_iter, prob_mut, constraint_eq, constraint_ueq, early_stop, n_processes=n_processes)
        self.func = func_w_kwargs_transformer(func, n_processes)
        self.func_kwargs = func_kwargs
        self.n_ones = n_ones

        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(self.n_dim)
        self.precision = np.array(precision) * np.ones(self.n_dim)  # works when precision is int, float, list or array

        # Lind is the num of genes of every variable of func（segments）
        Lind_raw = np.log2((self.ub - self.lb) / self.precision + 1)
        self.Lind = np.ceil(Lind_raw).astype(int)

        # if precision is integer:
        # if Lind_raw is integer, which means the number of all possible value is 2**n, no need to modify
        # if Lind_raw is decimal, we need ub_extend to make the number equal to 2**n,
        self.int_mode_ = (self.precision % 1 == 0) & (Lind_raw % 1 != 0)
        self.int_mode = np.any(self.int_mode_)
        if self.int_mode:
            self.ub_extend = np.where(self.int_mode_
                                      , self.lb + (np.exp2(self.Lind) - 1) * self.precision
                                      , self.ub)

        self.len_chrom = sum(self.Lind)

        self.crtbp()

    def fix(self, Chrom):
        sum_Chrom = np.sum(Chrom, axis=1)
        for i in np.where(sum_Chrom != self.n_ones)[0]:
            if sum_Chrom[i] < self.n_ones:
                idx_0 = np.where(Chrom[i, :] == 0)[0]
                idx_0 = np.random.permutation(idx_0)
                Chrom[i, idx_0[:self.n_ones - sum_Chrom[i]]] = 1
            elif sum_Chrom[i] > self.n_ones:
                idx_1 = np.where(Chrom[i, :] == 1)[0]
                idx_1 = np.random.permutation(idx_1)
                Chrom[i, idx_1[:sum_Chrom[i] - self.n_ones]] = 0
        sum_Chrom = np.sum(Chrom, axis=1)
        assert np.all(sum_Chrom == self.n_ones)
        return Chrom

    def crtbp(self):
        # create the population
        self.Chrom = np.random.randint(low=0, high=2, size=(self.size_pop, self.len_chrom))
        self.Chrom = self.fix(self.Chrom)
        return self.Chrom

    def gray2rv(self, gray_code):
        # Gray Code to real value: one piece of a whole chromosome
        # input is a 2-dimensional numpy array of 0 and 1.
        # output is a 1-dimensional numpy array which convert every row of input into a real number.
        _, len_gray_code = gray_code.shape
        b = gray_code.cumsum(axis=1) % 2
        mask = np.logspace(start=1, stop=len_gray_code, base=0.5, num=len_gray_code)
        return (b * mask).sum(axis=1) / mask.sum()

    def chrom2x(self, Chrom):
        cumsum_len_segment = self.Lind.cumsum()
        X = np.zeros(shape=(self.size_pop, self.n_dim))
        for i, j in enumerate(cumsum_len_segment):
            if i == 0:
                Chrom_temp = Chrom[:, :cumsum_len_segment[0]]
            else:
                Chrom_temp = Chrom[:, cumsum_len_segment[i - 1]:cumsum_len_segment[i]]
            X[:, i] = self.gray2rv(Chrom_temp)

        if self.int_mode:
            X = self.lb + (self.ub_extend - self.lb) * X
            X = np.where(X > self.ub, self.ub, X)
            # the ub may not obey precision, which is ok.
            # for example, if precision=2, lb=0, ub=5, then x can be 5
        else:
            X = self.lb + (self.ub - self.lb) * X
        return X

    def x2y(self):
        self.Y_raw = self.func(self.X, self.func_kwargs)
        if not self.has_constraint:
            self.Y = self.Y_raw
        else:
            # constraint
            penalty_eq = np.array([np.sum(np.abs([c_i(x) for c_i in self.constraint_eq])) for x in self.X])
            penalty_ueq = np.array([np.sum(np.abs([max(0, c_i(x)) for c_i in self.constraint_ueq])) for x in self.X])
            self.Y = self.Y_raw + 1e5 * penalty_eq + 1e5 * penalty_ueq
        return self.Y

    ranking = ranking.ranking
    selection = selection.selection_tournament_faster
    crossover = crossover.crossover_2point_bit
    mutation = mutation.mutation

    def to(self, device):
        '''
        use pytorch to get parallel performance
        '''
        try:
            import torch
            from sko.operators_gpu import crossover_gpu, mutation_gpu, selection_gpu, ranking_gpu
        except:
            print('pytorch is needed')
            return self

        self.device = device
        self.Chrom = torch.tensor(self.Chrom, device=device, dtype=torch.int8)

        def chrom2x(self, Chrom):
            '''
            We do not intend to make all operators as tensor,
            because objective function is probably not for pytorch
            '''
            Chrom = Chrom.cpu().numpy()
            cumsum_len_segment = self.Lind.cumsum()
            X = np.zeros(shape=(self.size_pop, self.n_dim))
            for i, j in enumerate(cumsum_len_segment):
                if i == 0:
                    Chrom_temp = Chrom[:, :cumsum_len_segment[0]]
                else:
                    Chrom_temp = Chrom[:, cumsum_len_segment[i - 1]:cumsum_len_segment[i]]
                X[:, i] = self.gray2rv(Chrom_temp)

            if self.int_mode:
                X = self.lb + (self.ub_extend - self.lb) * X
                X = np.where(X > self.ub, self.ub, X)
            else:
                X = self.lb + (self.ub - self.lb) * X
            return X

        self.register('mutation', mutation_gpu.mutation). \
            register('crossover', crossover_gpu.crossover_2point_bit). \
            register('chrom2x', chrom2x)

        return self

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        best = []
        bar = tqdm(range(self.max_iter), total=self.max_iter)
        for i in bar:
            self.X = self.chrom2x(self.Chrom)
            self.Y = self.x2y()
            self.ranking()
            self.selection()
            self.crossover()
            self.mutation()
            self.Chrom = self.fix(self.Chrom)

            # record the best ones
            generation_best_index = self.FitV.argmax()
            self.generation_best_X.append(self.X[generation_best_index, :])
            self.generation_best_Y.append(self.Y[generation_best_index])
            self.all_history_Y.append(self.Y)
            self.all_history_FitV.append(self.FitV)

            if self.early_stop:
                best.append(min(self.generation_best_Y))
                if len(best) >= self.early_stop:
                    if best.count(min(best)) == len(best):
                        break
                    else:
                        best.pop(0)
            bar.set_description(f'best_y: {self.generation_best_Y[-1]}')

        global_best_index = np.array(self.generation_best_Y).argmin()
        self.best_x = self.generation_best_X[global_best_index]
        self.best_y = self.func(np.array([self.best_x]), self.func_kwargs)
        return self.best_x, self.best_y

    fit = run

