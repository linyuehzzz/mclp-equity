import geopandas as gpd
from scipy.spatial.distance import cdist
import numpy as np
import math
import pickle

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize


###-----------coverage--------------###

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


# inequality: relative range
class MCLPE1(ElementwiseProblem):
    def __init__(self, w, a, p):
        super().__init__(n_var=a.shape[1], n_obj=2, n_ieq_constr=1, xl=0, xu=1, vtype=bool)
        self.w = w
        self.a = a
        self.p = p

    def _evaluate(self, x, out, *args, **kwargs):
        y = np.max(x * self.a, axis=1)

        # Objective 1
        obj1 = np.sum(np.sum(y * self.w.T, axis=1))

        # # Objective 2
        u_k = np.sum(y * self.w.T, axis=1) / np.sum(self.w, axis=0)
        u_bar = np.mean(u_k)
        obj2 = (np.max(u_k) - np.min(u_k)) / u_bar

        constr = np.sum(x) - self.p  # Constraint on the total number of facilities

        out["F"] = [-obj1, obj2]
        out["G"] = [constr]


# inequality: variance
class MCLPE2(ElementwiseProblem):
    def __init__(self, w, a, p):
        super().__init__(n_var=a.shape[1], n_obj=2, n_ieq_constr=1, xl=0, xu=1, vtype=bool)
        self.w = w
        self.a = a
        self.p = p

    def _evaluate(self, x, out, *args, **kwargs):
        y = np.max(x * self.a, axis=1)

        # Objective 1
        obj1 = np.sum(np.sum(y * self.w.T, axis=1))

        # # Objective 2
        u_k = np.sum(y * self.w.T, axis=1) / np.sum(self.w, axis=0)
        obj2 = np.var(u_k)

        constr = np.sum(x) - self.p  # Constraint on the total number of facilities

        out["F"] = [-obj1, obj2]
        out["G"] = [constr]


# inequality: theil index
class MCLPE3(ElementwiseProblem):
    def __init__(self, w, a, p):
        super().__init__(n_var=a.shape[1], n_obj=2, n_ieq_constr=1, xl=0, xu=1, vtype=bool)
        self.w = w
        self.a = a
        self.p = p

    def _evaluate(self, x, out, *args, **kwargs):
        y = np.max(x * self.a, axis=1)

        # Objective 1
        obj1 = np.sum(np.sum(y * self.w.T, axis=1))

        # # Objective 2
        u_k = np.sum(y * self.w.T, axis=1) / np.sum(self.w, axis=0)
        u_bar = np.mean(u_k)
        obj2 = 1 / self.w.shape[1] * np.sum(u_k / u_bar * np.log(u_k / u_bar))

        constr = np.sum(x) - self.p  # Constraint on the total number of facilities

        out["F"] = [-obj1, obj2]
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
s = 1000
t = 0.1

# read data
cook_centroids = gpd.read_file('data/cook_centroids_all.shp')
n1 = cook_centroids.shape[0]

# distance matrix
points = cook_centroids['geometry'].apply(lambda geom: (geom.x, geom.y)).tolist()
D = cdist(points, points, metric='euclidean')

# coverage
A = coverage_continuous(n1, D)

# attribute group
groups = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7']
n2 = len(groups)
W = cook_centroids[groups].to_numpy()

P = {
    1: MCLPE1,
    2: MCLPE2,
    3: MCLPE3
}
p_all = [60, 80, 100, 120, 140]


# main
with open('data/runtime.csv', 'w', newline='') as fw:
    fw.write('p,prob,runtime\n')
    fw.flush()

    # facility counts
    for p in p_all:
        # mclp
        problem = MCLP(W, A, p)

        algorithm = GA(
            pop_size=100,
            sampling=BinaryRandomSampling(),
            crossover=TwoPointCrossover(),
            mutation=BitflipMutation(),
            eliminate_duplicates=True
        )

        termination = get_termination("n_gen", 2000)

        res = minimize(problem,
                        algorithm,
                        termination,
                        seed=1,
                        save_history=True,
                        verbose=True)
        print("Best solution found: %s" % res.X.astype(int))
        print("Function value: %s" % res.F)
        print("Constraint violation: %s" % res.CV)

        F = np.concatenate([res.F, inequality(res.X.astype(int), W, A)])

        filename1 = 'data/obj/F_p' + str(p) + '.pickle'
        pickle.dump(F, open(filename1, "wb"))
        fw.write(str(p) + ',' + 'mclp' + ',' + str(res.exec_time) + '\n')
        fw.flush()

        # problem
        for prob in [1, 2, 3]:
            problem = P[prob](W, A, p)

            print(P[prob])

            algorithm = NSGA2(
                pop_size=100,
                sampling=BinaryRandomSampling(),
                crossover=TwoPointCrossover(),
                mutation=BitflipMutation(),
                eliminate_duplicates=True
            )

            termination = get_termination("n_gen", 2000)

            res = minimize(problem,
                            algorithm,
                            termination,
                            seed=1,
                            save_history=True,
                            verbose=True)
            print("Best solution found: %s" % res.X.astype(int))
            print("Function value: %s" % res.F)
            print("Constraint violation: %s" % res.CV)

            filename2 = 'data/obj/F_p' + str(p) + '_prob' + str(prob) + '.pickle'
            pickle.dump(res.F, open(filename2, "wb"))
            fw.write(str(p) + ',' + str(prob) + ',' + str(res.exec_time) + '\n')
            fw.flush()