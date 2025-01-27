from gurobipy import *
import numpy as np
import scipy.sparse as sp
from scipy.stats import norm, skewnorm
from scipy.linalg import sqrtm
from tqdm import tqdm

class ROSecurity:
    def __init__(self, schedule, n_slots, cost, n_hours_before_depature=4):
        self.N = schedule.shape[0]
        diag = schedule['Aircraft Capacity'].to_numpy()
        self.diag = np.tile(diag, 2)
        self.D = np.diag(self.diag)
        self.n_slots = n_slots
        self.cost = cost
        self.latest_arrival_time = schedule['slots'].to_numpy().astype(int)


        x_values = np.arange(0, n_hours_before_depature*60+1, 15)
        total_prob_per_block = []
        for i in range(len(x_values) - 1):
            total_prob = skewnorm.cdf(x_values[i + 1], 3, loc=93, scale=40) - skewnorm.cdf(x_values[i], 3, loc=93, scale=40)
            total_prob_per_block.append(total_prob)
        total_prob_per_block = np.array(total_prob_per_block)
        self.pmf = total_prob_per_block / np.sum(total_prob_per_block)


    def _compute_coef(self, j, flight_depature_time, alpha_it):
        prev = j - flight_depature_time
        return (1-alpha_it)*self.pmf[prev] * alpha_it


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

        base_matrix_first_row = [1] + [-i for i in self.pmf]
        base_matrix_second_row = [0] + [i for i in self.pmf]

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
                        quicksum(x[i,j]*self.diag[i]*self._compute_coef(j, self.latest_arrival_time[i], alpha[i,j]) for i in range(self.N) for j in range(self.latest_arrival_time[i]-16, self.latest_arrival_time[i]) if j >= 0)
                    )
                ),
            "robust_constraint_{t}",
            )

            x_mat = m.addMVar(((1+16)*self.N,), vtype=GRB.CONTINUOUS, name='x_mat')
            m.addConstrs((x_mat[i] == x[i,t] for i in range(self.N)))

            for j in range(16,0,-1):
                m.addConstrs(x_mat[self.N * (17-j) + i] == x[i, self.latest_arrival_time[i]-j] for i in range(self.N) if self.latest_arrival_time[i]-j >= 0)

            
            m.addConstr(
                (x_mat.T @ A_mat.T @ self.D.T @ sigma_matrix @ self.D @ A_mat @ x_mat <= rhs[t] * rhs[t])
            )

        for i in tqdm(range(self.N)):
            m.addConstr(quicksum(x[i,j] for j in range(self.n_slots)) == 1)
        for idx, time in enumerate(self.latest_arrival_time):
            m.addConstr(quicksum(x[idx,j] for j in range(int(time), int(self.n_slots))) == 0)

        for idx, time in enumerate(self.latest_arrival_time):
            m.addConstr(quicksum(x[idx,j] for j in range(self.n_slots) if (j < time-16) | (j >= time)) == 0)

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


