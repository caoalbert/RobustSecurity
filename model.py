from gurobipy import *
import numpy as np
import scipy.sparse as sp
from scipy.stats import norm
from scipy.linalg import sqrtm
from tqdm import tqdm

class ROSecurity:
    def __init__(self, schedule, n_slots, cost):
        self.N = schedule.shape[0]
        diag = schedule['Aircraft Capacity'].to_numpy()
        self.diag = np.tile(diag, 2)
        self.D = np.diag(self.diag)
        self.n_slots = n_slots
        self.cost = cost
        self.latest_arrival_time = schedule['slots'].to_numpy()
    
    def _calc_coef(self, j, alpha_it, n_neighbor, coef_only=False):
        if coef_only:
            return (n_neighbor+1-j) / (1+n_neighbor)*n_neighbor
        else:
            return (1-alpha_it) * (n_neighbor+1-j) / (1+n_neighbor)*n_neighbor


    def run(self, capacity, gamma, alpha, sigma, n_neighbor):
        # gamma is the confidence level
        # alpha is the rate of compliance
        # sigma is the standard deviation of the rate of compliance
        # n_neighbor is the number of neighbors (one-sided) that the assigned passengers can overflow to
        alpha_vector = np.array([alpha]*self.N)
        alpha = np.tile(alpha_vector, (self.n_slots, 1)).T

        sigma_matrix = np.zeros((2*self.N, 2*self.N))
        for i in range(self.N):
            sigma_matrix[i, i] = sigma

        base_matrix_first_row = [1] + [elem for elem in [-1 * self._calc_coef(j, None, n_neighbor, coef_only=True) for j in range(1, n_neighbor + 1)] for _ in range(2)]
        base_matrix_second_row = [0] + [elem for elem in [1 * self._calc_coef(j, None, n_neighbor, coef_only=True) for j in range(1, n_neighbor + 1)] for _ in range(2)]

        base_matrix = np.array([base_matrix_first_row, base_matrix_second_row])
        A_mat = sp.kron(base_matrix, sp.eye(self.N))
        self.base_matrix = base_matrix
        ppf_beta = norm.ppf(1-gamma)
        
    
        m = Model('RO Security')
        x = m.addVars(self.N, self.n_slots, vtype=GRB.CONTINUOUS, name='x')        
        rhs = m.addVars(self.n_slots, vtype=GRB.CONTINUOUS, name='rhs')

        for t in tqdm(range(self.n_slots)):
            m.addConstr(
                rhs[t] == 1/ppf_beta * (
                    capacity - ( 
                        quicksum(alpha[i,t]*self.diag[i]*x[i,t] for i in range(self.N)) + 
                        quicksum(self._calc_coef(j, alpha[i,t], n_neighbor)*self.diag[i]*x[i, t-j] for i in range(self.N) for j in range(1, n_neighbor+1) if t-j >= 0) +
                        quicksum(self._calc_coef(j, alpha[i,t], n_neighbor)*self.diag[i]*x[i, t+j] for i in range(self.N) for j in range(1, n_neighbor+1) if t+j < self.n_slots)
                    )
                ),
            "robust_constraint_{t}",
            )

            x_mat = m.addMVar(((1+2*n_neighbor)*self.N,), vtype=GRB.CONTINUOUS, name='x_mat')
            m.addConstrs((x_mat[i] == x[i,t] for i in range(self.N)))

            for j in range(1, n_neighbor+1):
                if t-j >= 0:
                    m.addConstrs(x_mat[self.N * (2*(j-1)+1) + i] == x[i, t-j] for i in range(self.N))
                else:
                    m.addConstrs(x_mat[self.N * (2*(j-1)+1) + i] == 0 for i in range(self.N))

                if t+j < self.n_slots:
                    m.addConstrs(x_mat[self.N * (2*(j-1)+2) + i] == x[i, t+j] for i in range(self.N))
                else:
                    m.addConstrs(x_mat[self.N * (2*(j-1)+2) + i] == 0 for i in range(self.N))
            
            m.addConstr(
                (x_mat.T @ A_mat.T @ self.D.T @ sigma_matrix @ self.D @ A_mat @ x_mat <= rhs[t] * rhs[t])
            )

        for i in tqdm(range(self.N)):
            m.addConstr(quicksum(x[i,j] for j in range(self.n_slots)) == 1)
        for idx, time in enumerate(self.latest_arrival_time):
            m.addConstr(quicksum(x[idx,j] for j in range(int(time), int(self.n_slots))) == 0)

        m.setObjective(quicksum(x[i,j]*self.cost[i,j] for i in range(self.N) for j in range(self.n_slots)), GRB.MINIMIZE)
        m.update()
        m.params.NonConvex = 2
        m.params.OutputFlag = 0
        m.optimize()
    


        solution = np.zeros((self.N, self.n_slots))
        for i in range(self.N):
            for j in range(self.n_slots):
                solution[i, j] = m.getVarByName(f'x[{i},{j}]').x

        return solution


