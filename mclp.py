import geopandas as gpd
from scipy.spatial.distance import cdist
import numpy as np
import math
import pickle

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize


###-----------coverage--------------###

# discrete coverage
def coverage_discrete(n1, D):
    A = np.zeros((n1, n1))
    for i in range(n1):
        for j in range(n1):
            if D[i, j] <= s:
                A[i, j] = 1
    return A

# continuous coverage
def coverage_continuous(n1, D):
    A = np.zeros((n1, n1))
    for i in range(n1):
        for j in range(n1):
            A[i, j] = math.exp(-t * D[i, j] / s)
    return A


###-----------problems--------------###

# mclp
class MCLP(ElementwiseProblem):
    def __init__(self, w, a, p):
        super().__init__(n_var=a.shape[1], n_obj=1, n_ieq_constr=1, xl=0, xu=1, vtype=bool)
        self.w = w
        self.a = a
        self.p = p

    def _evaluate(self, x, out, *args, **kwargs):
        y = np.max(x * self.a, axis=1)

        # Objective 1
        obj1 = np.sum(np.sum(y * self.w.T, axis=1))

        constr = np.sum(x) - self.p  # Constraint on the total number of facilities

        out["F"] = [-obj1]
        out["G"] = [constr]


###-----------inequality--------###

def inequality(x, W, A):
    y = np.max(x * A, axis=1)

    # relative range
    u_k = np.sum(y * W.T, axis=1) / np.sum(W, axis=0)
    u_bar = np.mean(u_k)
    e1 = (np.max(u_k) - np.min(u_k)) / u_bar

    # variance
    e2 = np.var(u_k)

    # theil index
    e3 = 1 / W.shape[1] * np.sum(u_k / u_bar * np.log(u_k / u_bar))

    return [e1, e2, e3]


###-----------main--------------###

# user-defined parameters
p = 100
s = 1000
t = 2

# read data
cook_centroids = gpd.read_file('data/cook_centroids_all.shp')
n1 = cook_centroids.shape[0]

# distance matrix
points = cook_centroids['geometry'].apply(lambda geom: (geom.x, geom.y)).tolist()
D = cdist(points, points, metric='euclidean')

C = {
    1: coverage_discrete,
    2: coverage_continuous
}
G = [['S1', 'S2'], 
     ['A1', 'A2', 'A3'],
     ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7'],
     ['HHI1', 'HHI2', 'HHI3']]

# main
for cov in [1, 2]:
    # coverage
    A = C[cov](n1, D)
    
    # attribute group
    for g_idx, groups in enumerate(G):
        n2 = len(groups)
        W = cook_centroids[groups].to_numpy()

        problem = MCLP(W, A, p)

        algorithm = GA(
            pop_size=100,
            sampling=BinaryRandomSampling(),
            crossover=TwoPointCrossover(),
            mutation=BitflipMutation(),
            eliminate_duplicates=True
        )

        termination = get_termination("n_gen", 1000)

        res = minimize(problem,
                        algorithm,
                        termination,
                        seed=1,
                        save_history=True,
                        verbose=True)
        
        # convergence
        n_evals = [algo.evaluator.n_eval for algo in res.history]
        hist_cv_avg = []
        for algo in res.history:
            opt = algo.opt
            hist_cv_avg.append(algo.pop.get("CV").mean())
            feas = np.where(opt.get("feasible"))[0]
        k = np.where(np.array(hist_cv_avg) <= 0.0)[0].min()
        print(f"Whole population feasible in Generation {k} after {n_evals[k]} evaluations.")

        F = np.concatenate([res.F, inequality(res.X.astype(int), W, A)])

        filename1 = 'data/obj/F_cov' + str(cov) + '_grp' + str(g_idx) + '.pickle'
        pickle.dump(F, open(filename1, "wb"))
        filename2 = 'data/sols/X_cov' + str(cov) + '.pickle'
        pickle.dump(res.X.astype(int), open(filename2, "wb"))