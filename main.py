#%%
import os
import sys
import pickle
import numpy as np
sys.path.append('./helper_functions')
from useful_chunk import create_dir
from control import dare
import matplotlib.pyplot as plt
from numpy.linalg import inv
from termcolor import colored # to color the error information
from joblib import Parallel, delayed
import scipy.stats as stats
from coverage_regret_helper import get_K_grad
import seaborn as sns
from sklearn.linear_model import LinearRegression

# plot setup
from matplotlib import rc
font = {'size'   : 32}
params = {'axes.labelsize': 32,
          'axes.titlesize': 32,
          'text.usetex':True,
          'text.latex.preamble':r'\usepackage{amsmath,amsfonts,amssymb}'}
plt.rcParams.update(params)
rc('font', **font)
rc('text', usetex=True)

# set random seed
np.random.seed(1234)
print("Loading package success!")



# %%
'''
A dictionary which stores all the parameters in the system
x_{t+1} = Ax_t + Bu_t + \varepsilon_t
u_t = Khat u_t + \eta_t
system noise -- \varepsilon_t \sim Normal(0, \std^2 I_d)
exploration noise -- \eta_t \sim Normal(0, t^{1-beta}\log^alpha(t)tau^2 I_m)

The LQR loss is defined by \sum_{t=1}^\top x_t^\top Q x_t + u_t^\top R u_t
d: dimension of the state x_t
m: dimension of the input u_t
Q: weight matrix used to define the norm of x_t in LQR loss
R: weight matrix used to define the norm of u_t in LQR loss
A: linear transformation matrix connecting previous state x_t 
    and current state x_{t+1}
B: linear transformation matrix connecting previous input u_t 
    and current state x_{t+1}
K_0: the safety backup we use in our algorithm
C_K: the threshold on ||K||, if ||Khat|| > C_K, then use K_0 instead
C_x: the threshold on ||K||, if ||Khat|| > C_x, then use K_0 instead
alpha: the log factor in exploration noise 
name: folder name for storing the result
std: the system noise std 
tau: the std scale in exploration noise
S_0: the initial place x_0
'''
# stable system
stable_dict = {
    'd': 2, 
    'm': 1,
    'Q': np.eye(2),
    'R': np.eye(1),
    # 'N': np.zeros([2, 1]),
    'A': np.array([[0.8, 0.1], [0, 0.8]]),
    'B': np.array([0, 1]).reshape(2, 1),
    # A_hat = np.array([[0.7, 0.3], [0.2, 0.7]])
    # B_hat = np.array([0.3, 0.7]).reshape(2, 1)
    'K_0': np.array([0, 0]).reshape(1,2),
    'C_K': 5,
    'C_x': 1,
    # 'alpha': 2,
    'name': "stable_system",
    # the std of system error \varepsilon_t
    'std': 1, 
    # the std of input exploration noise
    # which is: \eta_t \sim \calN(0, t^{1-beta}\log^alpha(t)tau^2 I_m)
    'tau': 0.2,
    'S_0': np.zeros(2) # the initial place x_0
}


# %%
'''
the class LQR contains all parameter choices
remember to replace these after coding finisihed

new: old
T: self.n
repeats: test_num
self.n: self.d
self.d: self.m
'''
class LQR:
    def __init__(self, para_dict, n, test_num, beta, alpha, log_flag = False, copy = None):
        '''
        para_dict: dictionary with all parameter values
        self.n: the one shot experiment time length
        test_num: repeat the same one shot experiment for 'test_num' times
        beta, alpha: choices in the decay rate of exploration noise eta_t
        '''
        # check the beta condition
        if beta < 0:
            sys.exit(colored("Beta can't be nagative value!", "red"))

        self.d = para_dict['d']
        self.m = para_dict['m']
        self.md = self.m + self.d # the total dimension of state and input
        self.Q = para_dict['Q']
        self.R = para_dict['R']
        self.A = para_dict['A']
        self.B = para_dict['B']
        self.K_0 = para_dict['K_0']
        # solve P_0 by K_0

        self.C_K = para_dict['C_K']
        self.C_x = para_dict['C_x']
        self.log_flag = log_flag

        # name is unique identifiy of the object
        # name is also used in storing intermediate results
        self.name = f"{para_dict['name']}_beta{beta}_alpha{alpha}"

        self.n = n
        self.test_num = test_num
        self.std = para_dict['std']
        self.tau = para_dict['tau']
        self.S_0 = para_dict['S_0']

        # assume we have ridge regression penalty 1e-5
        # this can ensure a unique solution even when the cov matrix is singular
        self.lamda = 1e-5 

        self.beta = beta
        self.alpha = alpha


        # parameters derived from given ones
        '''
        the K defined in our paper is in the opposite sign of 
        the usual solution from DARE equations 
        DARE: discrete-time algebraic Riccati equation 
        ref: https://en.wikipedia.org/wiki/Algebraic_Riccati_equation
        '''
        self.P, _, self.K = dare(self.A, self.B, self.Q, self.R)
        self.K = np.array(-self.K).reshape(self.m, self.d)
        self.A_BK = self.A + self.B @ self.K

        # the average regret in Theorem 1 is:
        # tau^2/beta Tr(B^TPB+R)T^{\beta-1}log^\alpha(T)
        # total regret is:
        # tau^2/beta Tr(B^TPB+R)T^{\beta}log^\alpha(T)
        self.theory_regret_coef = np.trace(self.B.T.dot(self.P).dot(self.B) + self.R)


        
        # create a dir if it not exist
        # this dir is used to store results generated from this class
        create_dir("./{}".format(self.name))
        print("This dir is used to store results generated from this LQR class")

        

        # assign memory to store all intermediate results needed all from {0,1,...,T}
        self.state_testnum = np.zeros(
            [self.test_num, self.n + 1, self.d])
        self.state_star_testnum = np.zeros(
            [self.test_num, self.n + 1, self.d])
        self.input_testnum = np.zeros(
            [self.test_num, self.n + 1, self.m])
        self.input_star_testnum = np.zeros(
            [self.test_num, self.n + 1, self.m])
        self.cov_matrix_testnum = np.zeros(
            [self.test_num, self.n + 1, self.md, self.md])
        for i in range(self.test_num): # initialize
            self.cov_matrix_testnum[i, :2] = self.lamda * np.eye(self.md)  
        self.inv_cov_testnum = np.zeros(
            [self.test_num, self.n + 1, self.md, self.md])
        self.XY_sum_testnum = np.zeros(
            [self.test_num, self.n + 1, self.md, self.d])
        # only up to time T(self.n), because the AB_hat subscript is t-1 instead of t at timestep t
        self.AB_hat_testnum = np.zeros(
            [self.test_num, self.n, self.d, self.md])
        for i in range(self.test_num): # initialize None because no estimate in the first one step
            self.AB_hat_testnum[i, 0] = None
        

        # the direct K estimate from Ahat Bhat
        self.K_tilde_testnum = np.zeros(
            [self.test_num, self.n + 1, self.m, self.d])
        for i in range(self.test_num): # initialize None because no estimate in the first two steps
            self.K_tilde_testnum[i, :2] = None

        # the true controller, sometimes is K_tilde, somoetimes is K_0
        self.K_hat_testnum = np.zeros(
            [self.test_num, self.n + 1, self.m, self.d])
        for i in range(self.test_num): # initialize
            self.K_hat_testnum[i, :2] = self.K_0

        # the direct P estimate from  Ahat Bhat
        self.P_tilde_testnum = np.zeros(
            [self.test_num, self.n + 1, self.d, self.d])
        for i in range(self.test_num): # initialize None because no estimate in the first two steps
            self.P_tilde_testnum[i, :2] = None
        
        self.hit_C_x_index = [[] for _ in range(self.test_num)] # a list keeping index of K_0 induced by hitting C_x
        self.hit_C_K_index = [[] for _ in range(self.test_num)] # a list keeping index of K_0 induced by hitting C_K

        # the only two random components we use in our algorithm
        if log_flag:
            self.noise_eta = copy.noise_eta
            self.noise_varepsilon = copy.noise_varepsilon
        else:
            # system noise t= {0,1...,T-1}, deciding x_t {1,2...,T}
            self.noise_varepsilon = self.std * np.random.randn(self.test_num, self.n, self.d)

            # input noise t={0,1...,T}, deciding u_t {0,1...,T}
            self.noise_eta = self.tau * np.random.randn(self.test_num, self.n + 1, self.m)
            eta_coef = np.ones(self.n + 1)
            # notice that the coef is actually not decreasing for a long time because of the log factor..
            for i in range(2, self.n + 1):
                eta_coef[i] = (i ** ((beta - 1) / 2)) * (np.log(i) ** (self.alpha / 2))
            self.noise_eta = np.einsum('n, tnm->tnm', eta_coef, self.noise_eta)

            # print a success message upon finiishing __init__ function
            print('Initialize success!')

    # u_t = Khat u_t + \eta_t  
    def next_input(self, Khat, cur_state, input_noise):
        return np.dot(Khat, cur_state) + input_noise

    # x_{t+1} = Ax_t + Bu_t + \varepsilon_t
    def next_state(self, cur_state, cur_input, system_noise):
        return np.dot(self.A, cur_state) + np.dot(self.B, cur_input) + system_noise


    # def update_the_system(self, state_list, input_list, system_noise_list, input_noise_list, cur_cov, t):
    def update_the_system(self, run, t):
        '''
        update the whole system at time t by loop, notice that this only works when time t >= 2
        '''
        # update the cov matrix
        S, U = self.state_testnum[run, t-2], self.input_testnum[run, t-2]
        Y = self.state_testnum[run, t-1]
        X = np.concatenate((S, U))
        self.cov_matrix_testnum[run, t] = cov = self.cov_matrix_testnum[run, t-1] + np.outer(X, X)
        self.inv_cov_testnum[run, t] = inv_cov = inv(cov)
        
        # calculate the new estimates Ahat_{t-1} Bhat_{t-1}
        self.XY_sum_testnum[run, t] = self.XY_sum_testnum[run, t-1] + np.outer(X, Y)

        if self.log_flag and (t & (t-1) != 0):
            self.AB_hat_testnum[run, t-1] = self.AB_hat_testnum[run, t-2]
            self.K_tilde_testnum[run, t] = self.K_tilde_testnum[run, t-1]
            self.P_tilde_testnum[run, t] = self.P_tilde_testnum[run, t-1]
        else:
            self.AB_hat_testnum[run, t-1] = np.dot(inv_cov, self.XY_sum_testnum[run, t]).T
            A_hat = self.AB_hat_testnum[run, t-1][:, :self.d].reshape(self.d, self.d)
            B_hat = self.AB_hat_testnum[run, t-1][:, self.d:].reshape(self.d, self.m)

            # get Ktilde, Ptilde from Ahat_{t-1} Bhat_{t-1}
            try:
                Ptilde, _, Ktilde = dare(A_hat, B_hat, self.Q, self.R)
                self.K_tilde_testnum[run, t] = -Ktilde
                self.P_tilde_testnum[run, t] = Ptilde
            except:
                print(colored("err when solve dare", "red"))
                self.K_tilde_testnum[run, t] = self.K_0
                print(A_hat, B_hat)
                self.P_tilde_testnum[run, t] = None

        # new S[t]
        self.state_testnum[run, t] = self.next_state(self.state_testnum[run, t-1], 
            self.input_testnum[run, t-1], self.noise_varepsilon[run, t-1])

        # new S_star[t]
        self.state_star_testnum[run, t] = self.next_state(self.state_star_testnum[run, t-1], 
            self.input_star_testnum[run, t-1], self.noise_varepsilon[run, t-1])

        
        # decide to whether use K_0 
        C_x_flag = np.linalg.norm(self.state_testnum[run, t]) > self.C_x * np.log(t)
        C_K_flag = np.linalg.norm(self.K_tilde_testnum[run, t]) > self.C_K
        if C_x_flag or C_K_flag: # use safety
            if C_x_flag:
                self.hit_C_x_index.append(t)
            if C_K_flag:
                self.hit_C_K_index.append(t)
            self.K_hat_testnum[run, t] = self.K_0
        else:
            self.K_hat_testnum[run, t] = self.K_tilde_testnum[run, t]

        # new U[t]
        self.input_testnum[run, t] = self.next_input(
            self.K_hat_testnum[run, t], self.state_testnum[run, t], self.noise_eta[run, t])

        # new U_star[t]
        self.input_star_testnum[run, t] = self.next_input(
            self.K, self.state_star_testnum[run, t], self.noise_eta[run, t])


        return 

    # the first input and second input all use K_0 as controller
    def initial_steps(self, S, U, run, setting = "myalg"):
        eta = self.noise_eta[run]
        epsilon = self.noise_varepsilon[run]
        U[0] = self.next_input(self.K_0, S[0], eta[0])
        S[1] = self.next_state(S[0], U[0], epsilon[0])
        U[1] = self.next_input(self.K_0, S[1], eta[1])

        # do a sanity check
        if any(S[0] == 0):
            if not (any(np.round(eta[0] - U[0], 4) == 0) and any(np.round(epsilon[0] + self.B.dot(U[0]) - S[1], 4) == 0)):
                print('U[0]', eta[0], U[0])
                print('S[1]', epsilon[0] + self.B.dot(U[0]), S[1])
                print('U[1]', U[1])
                sys.exit(colored("Please check the initial steps in states and inputs!", "red"))
        
        if setting == "myalg":
            X = np.concatenate((S[0], U[0]))
            self.cov_matrix_testnum[run, 0] = self.cov_matrix_testnum[run, 0] + np.outer(X, X)
            X = np.concatenate((S[1], U[1]))
            self.cov_matrix_testnum[run, 1] = self.cov_matrix_testnum[run, 0] + np.outer(X, X)
            self.inv_cov_testnum[run, 0] = inv(self.cov_matrix_testnum[run, 0])
            self.inv_cov_testnum[run, 1] = inv(self.cov_matrix_testnum[run, 1])



    # run one single trajectory of experiment from t=0 to t=T
    def run_once(self, run):
        # initial steps in both our algorithm and optimal algorithm
        S = self.state_testnum[run]
        U = self.input_testnum[run]
        self.initial_steps(S, U, run) 

        S = self.state_star_testnum[run]
        U = self.input_star_testnum[run]
        self.initial_steps(S, U, run)

        
        # estimate A, B, K starting from step t=2
        for i in range(2, self.n + 1):
            self.update_the_system(run, t=i)  
                

        return


    def run_all(self):
        '''
        Repeat the run_once function for self.test_num times.
        i.e. repeat the experiments to collect enough data 
            in order to draw valid statistical conclusions.
        '''
        # one can make this parallel
        for i in range(self.test_num):
            if i % 10 == 0:
                print(f"{i+1}-th experiment finished, total {self.test_num}")
            self.run_once(run = i)
        



'''
test the class
'''
# stable_LQR = LQR(para_dict = stable_dict, n = 1000, test_num = 1000, beta = 0.5, alpha = 2)
# stable_LQR.run_all()
# stable_LQR.state_star_testnum[:, :10]
# stable_LQR.data_for_plot()

stable_LQR_log = LQR(para_dict = stable_dict, n = 1000, test_num = 1000, beta = 0.5, alpha = 2,
     log_flag=True, copy = stable_LQR)
stable_LQR_log.run_all()


# %%
'''
test some values 
'''
# stable_LQR.AB_hat_testnum
# stable_LQR.cov_matrix_testnum
# stable_LQR.inv_cov_testnum
# stable_LQR.XY_sum_testnum
# stable_LQR.K_hat_testnum[0, :]
# stable_LQR.K_tilde_testnum[0, :10]
# stable_LQR.P_tilde_testnum[0, :10]


# %%
'''
save the object
'''
# with open(f'{stable_LQR.name}/n-{stable_LQR.n}test_num-{stable_LQR.test_num}.pkl', 'wb') as file:
#     pickle.dump(stable_LQR, file)

with open(f'{stable_LQR.name}/n-{stable_LQR.n}test_num-{stable_LQR.test_num}-log.pkl', 'wb') as file:
    pickle.dump(stable_LQR_log, file)

# ----------------- can restart from here -------------------
# %%
'''open the object'''
# with open(f'stable_system_beta0.5_alpha2/n-200test_num-100.pkl', 'rb') as file:
#     stable_LQR = pickle.load(file)

# stable_LQR.AB_hat_testnum[:10, :10]
# %%


def LQR_cost(state_array, input_array, Q, R):
    '''
    LQR cost, notice that we don't count in the cost at time 0
    '''
    cost_step = np.einsum('rnj,rnj->rn', np.einsum('rni,ij->rnj', state_array, Q), 
        state_array) + np.einsum('rnj,rnj->rn', 
        np.einsum('rni,ij->rnj', input_array, R), input_array) 
    cost_cumsum = np.cumsum(cost_step, axis = 1)
    return cost_cumsum

class SummarizeLQR():
    '''
    a new class to summarize data
    required summary statistics:
    Advantage versus logarithmic update: log_minus_control
    Regret Ratio: 
        regret_ratio_para; t \in {1, 2, ..., t}
        regret_ratio_ob; t \in {2, ..., t}
    Differing Convergence Rates (L2 norm): 
        ABK_err, AB_err; t \in {0, ..., t-1}
        K_err; t \in {0, ..., t}
    Confidence Region Coverage: 
        coverage_ellipsoid, coverage_safety, coverage_ellipsoid_K; t \in {1, 2, ..., t-1}
    Prediction Region Coverage: 
        coverage_safety_naive, coverage_safety_our; t \in {1, 2, ..., t-1}
    '''
    def __init__(self, LQR_obj: LQR, LQR_obj_log = None):
        self.A = LQR_obj.A
        self.B = LQR_obj.B
        self.n = LQR_obj.n
        self.d = LQR_obj.d
        self.m = LQR_obj.m
        self.md = LQR_obj.md
        self.K = LQR_obj.K
        self.Q = LQR_obj.Q
        self.R = LQR_obj.R
        self.A_BK = self.A + self.B @ self.K
        self.P = LQR_obj.P
        self.tau = LQR_obj.tau
        self.std = LQR_obj.std
        self.beta = LQR_obj.beta
        self.alpha = LQR_obj.alpha
        self.test_num = LQR_obj.test_num
        self.name = LQR_obj.name

        # the cost starts from time t = 1
        cost_myalg = \
            LQR_cost(LQR_obj.state_testnum[:,1:], LQR_obj.input_testnum[:,1:], self.Q, self.R)
        cost_optimal = \
            LQR_cost(LQR_obj.state_star_testnum[:,1:], LQR_obj.input_star_testnum[:,1:], self.Q, self.R)
        
        # cost difference between our alg and optimal solution u_t = Kx_t
        regret = cost_myalg - cost_optimal

        

        if LQR_obj_log:
            self.log_minus_flag = True
            cost_myalg_log = \
                LQR_cost(LQR_obj_log.state_testnum[:,1:], LQR_obj.input_testnum[:,1:], self.Q, self.R)
            # stepwise update advantage in regret, start from t = 1
            log_minus_control_cost =  cost_myalg_log - cost_myalg
            self.log_minus_control_cost_UB = np.quantile(log_minus_control_cost, q=0.95, axis=0) # store this 
            self.log_minus_control_cost_LB = np.quantile(log_minus_control_cost, q=0.05, axis=0) # store this 
            self.log_minus_control_cost_median = np.quantile(log_minus_control_cost, q=0.5, axis=0) # store this 
            self.log_minus_control_cost_mean = np.mean(log_minus_control_cost,axis=0)  
            # print(self.log_minus_control_cost_mean)
        else:
            self.log_minus_flag = False
        


        #         A_hat = 
        #         self.AB_hat_testnum = np.zeros(
        #         [self.test_num, self.n, self.d, self.md])
        def decompose_AB_hat(LQR_obj):
            A_hat = LQR_obj.AB_hat_testnum[:, :, :, :self.d]
            if (self.m == 1):
                B_hat = LQR_obj.AB_hat_testnum[:, :, :, None,-1] # (6, 2000, 2, 1)
            else:
                B_hat = LQR_obj.AB_hat_testnum[:, :, :, self.d:]
            return A_hat, B_hat
        
        A_hat, B_hat = decompose_AB_hat(LQR_obj)
        P_tilde = LQR_obj.P_tilde_testnum

        # print(np.round(regret))

        # the regret in theory is Tr(B'PB + R) * [Sum of eta_t variances]
        eta_coef = np.ones(self.n)* (self.tau ** 2)
        for i in range(2, self.n + 1):
            eta_coef[i-1] = (i ** (self.beta - 1) ) * (np.log(i) ** self.alpha)
        sum_eta_t_variance = np.cumsum(eta_coef) * (self.tau ** 2)
        main_term_regret = np.trace(self.B.T @ self.P @ self.B + self.R) * sum_eta_t_variance 
        # print(main_term_regret.shape)
        # print(np.round(main_term_regret))

        # parameter version ratio, start from t = 1
        regret_ratio_para = regret/main_term_regret
        self.regret_ratio_para_UB = np.quantile(regret_ratio_para, q=0.95, axis=0)
        self.regret_ratio_para_LB = np.quantile(regret_ratio_para, q=0.05, axis=0)
        self.regret_ratio_para_mean = np.mean(regret_ratio_para, axis=0)
        self.regret_ratio_para_median = np.quantile(regret_ratio_para, q=0.5, axis=0)

        # B_hat and P_tilde only has estimate from time t = 2
        # print(B_hat[:,1:].shape, P_tilde[:,2:].shape)
        tmp = np.einsum('rnud, rndm->rnum', 
            np.einsum('rnxu, rnxd->rnud', B_hat[:,1:], P_tilde[:,2:]), B_hat[:,1:]) + self.R
        tmp = np.trace(tmp, axis1=2, axis2=3)
        # print(tmp.shape, sum_eta_t_variance[2:].shape)
        estimated_main_term_regret = np.einsum('rn, n->rn', tmp, sum_eta_t_variance[1:])

        # the ratio starts from t = 2, while regret starts from t = 1
        regret_ratio_ob = regret[:,1:]/estimated_main_term_regret
        self.regret_ratio_ob_UB = np.quantile(regret_ratio_ob, q=0.95, axis=0)
        self.regret_ratio_ob_LB = np.quantile(regret_ratio_ob, q=0.05, axis=0)
        self.regret_ratio_ob_median =np.quantile(regret_ratio_ob, q=0.5, axis=0)
        self.regret_ratio_ob_mean =np.mean(regret_ratio_ob, axis=0)

        # ABK_err, AB_err, K_err (L2 norm)
        oracle_AB = np.concatenate((self.A, self.B), axis=1)
        AB_err = LQR_obj.AB_hat_testnum - oracle_AB
        AB_err_abs = np.abs(AB_err)
        # print(LQR_obj.AB_hat_testnum.shape)
        self.AB_err_L2_UB = np.quantile(np.linalg.norm(AB_err_abs, axis=(2,3), ord='fro'), 0.95, axis=0)
        self.AB_err_L2_LB = np.quantile(np.linalg.norm(AB_err_abs, axis=(2,3), ord='fro'), 0.05, axis=0)
        self.AB_err_L2_mean = np.mean(np.linalg.norm(AB_err_abs, axis=(2,3), ord='fro'), axis=0)
        # print(AB_err_L2_mean)

        # notice that some K_hat is actually K_0 
        K_err = LQR_obj.K_hat_testnum - self.K
        K_err_abs = np.abs(K_err)
        self.K_err_L2_UB = np.quantile(np.linalg.norm(K_err_abs, axis=(2,3), ord='fro'), 0.95, axis=0)
        self.K_err_L2_LB = np.quantile(np.linalg.norm(K_err_abs, axis=(2,3), ord='fro'), 0.05, axis=0)
        self.K_err_L2_mean = np.mean(np.linalg.norm(K_err_abs, axis=(2,3), ord='fro'), axis=0)
        # print(K_err_L2_mean)

        tmp = LQR_obj.AB_hat_testnum - oracle_AB
        if (self.m == 1):
            ABK_err = tmp[:,:,:,:-1] + tmp[:,:,:,None,-1].dot(self.K)
        else:
            ABK_err = tmp[:,:,:,:self.d] + tmp[:,:,:,self.d:].dot(self.K)
        self.ABK_err_L2_UB = np.quantile(np.linalg.norm(np.abs(ABK_err), axis=(2,3), ord='fro'), 0.95, axis=0)
        self.ABK_err_L2_LB = np.quantile(np.linalg.norm(np.abs(ABK_err), axis=(2,3), ord='fro'), 0.05, axis=0)
        self.ABK_err_L2_mean = np.mean(np.linalg.norm(np.abs(ABK_err), axis=(2,3), ord='fro'), axis=0)

        '''
        ellipsoid coverage: only avaiable from t \in {1, 2, ..., t-1}
        '''
        tmp = 0
        for i in range(self.d):
            AB_err_0 = AB_err[:,:,i,:]
            tmp1 = np.einsum('rnj, rnjk->rnk', AB_err_0[:,1:], LQR_obj.cov_matrix_testnum[:,2:])
            tmp1 = np.einsum('rnj, rnj->rn', tmp1, AB_err_0[:,1:])

            tmp = tmp + tmp1

        tmp_UB = stats.chi2.ppf(0.95, self.d*self.md)

        # coverage_flag_ = (tmp < tmp_UB) & (tmp > tmp_LB)
        coverage_flag_ = tmp < tmp_UB
        self.coverage_ellipsoid = np.mean(coverage_flag_, axis=0)  
        # print(tmp_UB)
        # print(tmp)

        '''
        safety gaurantee: coverage_safety
        '''
        # E[pred_err] = xhat_{t+1} - E[x_{t+1}]= (Theta_{t}-Theta)z_t
        # (sigma^2 z_t invcov_{t-1} z_t)^{-1/2} E[pred_err]
        # t \in {1, 2, ..., t-1}
        z_ = np.concatenate((LQR_obj.state_testnum, LQR_obj.input_testnum), axis=2)[:,1:-1]
        inv_cov = LQR_obj.inv_cov_testnum[:,:-2] # t \in {0, 1, ..., t-2}; one step smaller than z
        AB_err = AB_err[:,1:]  # t \in {1, 2, ..., t-1} same as z
        # cov_ = LQR_obj.cov_matrix_testnum[:,:-2]

        pred_var_tmp = np.einsum('rnz, rnzy->rny', z_, inv_cov)
        pred_var_tmp = np.einsum('rny, rny->rn', pred_var_tmp, z_)
        pred_var_tmp = 1/pred_var_tmp
        # print(pred_var_tmp.shape)
        # print(pred_var_tmp)

        # t \in {1, 2, ..., t-1}
        pred_err = np.einsum('rndz, rnz->rnd', AB_err, z_)
        pred_err_norm_square = np.einsum('rnd, rnd->rn', pred_err, pred_err)
        safety_test_value = pred_var_tmp * pred_err_norm_square
        tmp_UB = stats.chi2.ppf(0.95, self.d)
        coverage_flag_ = safety_test_value < tmp_UB
        self.coverage_safety = np.mean(coverage_flag_, axis=0)
        # print(tmp_UB)
        # print(np.round(safety_test_value, 1)) 
        # print(coverage_flag_, coverage_safety)

        '''
            coverage_ellipsoid_K
        '''
        def get_coverage_prop(i, LQR_obj):
            if(i % 10 == 0):
                print(i)
            coverage_sum = 0
            K_test_value_array = np.zeros(self.test_num)
            for run in range(self.test_num):
                A_hat, B_hat = decompose_AB_hat(LQR_obj)
                A_hat = A_hat[run,i]
                B_hat = B_hat[run,i]
                P_hat = LQR_obj.P_tilde_testnum[run,i+1]
                K_tilde = LQR_obj.K_tilde_testnum[run,i+1]
                cov = LQR_obj.cov_matrix_testnum[run,i]
                inv_cov = LQR_obj.inv_cov_testnum[run,i]
                K_grad = get_K_grad(A_hat, B_hat, K_tilde, P_hat, self.R) # shape md, d(d+m)
                AB_cov = np.kron(np.eye(self.d), inv_cov)
                # print(AB_cov)
                K_cov = K_grad.dot(AB_cov).dot(K_grad.T)
                K_tilde_vec = (K_tilde-self.K).reshape(-1)
                K_test_value = K_tilde_vec.dot(inv(K_cov)).dot(K_tilde_vec)
                K_test_value_array[run] = K_test_value
                chi2_UB = stats.chi2.ppf(0.95, self.d*self.m)
                coverage_flag = K_test_value < chi2_UB
                coverage_sum += coverage_flag
            # coverage_prop_[i] = coverage_sum/self.test_num
            if (i % 10 == 0):
                print(np.mean(K_test_value_array))
                print(np.median(K_test_value_array))
            return coverage_sum/self.test_num

        '''
        this step may take a long time
        '''
        # coverage_result = Parallel(n_jobs=-1)(delayed(get_coverage_prop)
        #                     (i, LQR_obj) for i in range(1, self.n))

        # self.coverage_ellipsoid_K = np.array(coverage_result)
        # print(self.coverage_ellipsoid_K.shape)

        # print(coverage_ellipsoid_K)

        '''
        coverage_safety_naive, coverage_safety_our
        '''
        # our prediction region coverage t \in {2, 3, ..., t-1}
        pred = np.einsum('rndz, rnz->rnd', LQR_obj.AB_hat_testnum[:,1:], z_)
        pred_err = z_[:,1:,:self.d] - pred[:,:-1]
        pred_err_norm_true = np.einsum('rnd, rnd->rn', pred_err, pred_err)
        # print(np.round(pred_err_norm_true[:,-1], 1))
        safety_test_value = 1/(1/pred_var_tmp[:,:-1] + 1) * pred_err_norm_true
        tmp_UB = stats.chi2.ppf(0.95, self.d)
        coverage_flag_ = safety_test_value < tmp_UB
        # print(tmp_UB)
        # print(np.round(safety_test_value, 1))
        self.coverage_safety_our = np.mean(coverage_flag_, axis=0)
        # print(self.coverage_safety_our)

        # the naive prediction region coverage
        safety_test_value = pred_err_norm_true
        tmp_UB = stats.chi2.ppf(0.95, self.d)
        coverage_flag_ = safety_test_value < tmp_UB
        # print(np.round(safety_test_value, 1))
        self.coverage_safety_naive = np.mean(coverage_flag_, axis=0)
        # print(self.coverage_safety_naive)



stable_LQR_summary = SummarizeLQR(stable_LQR, stable_LQR_log)
# SummarizeLQR(stable_LQR)
# %%
'''
save the summary
'''
with open(f'{stable_LQR.name}/n-{stable_LQR.n}test_num-{stable_LQR.test_num}-summary.pkl', 'wb') as file:
    pickle.dump(stable_LQR_summary, file)
# stable_LQR_summary.plot()
# %%
def plot(LQR_summary: SummarizeLQR):
    # fig saving path
    path = f'{LQR_summary.name}/n-{LQR_summary.n}test_num-{LQR_summary.test_num}-fig'
    create_dir(path)
    n = LQR_summary.n

    # Benefit of stepwise update t \in {1, 2, ..., t}
    if LQR_summary.log_minus_flag:
        cutoff = 1
        clrs = sns.color_palette("husl", 10)
        fig, axs = plt.subplots(1, 1, figsize=(12, 8))
        with sns.axes_style("darkgrid"): 
            axs.axhline(0, c="black", linewidth = 2)
            axs.plot(range(1,n+1), LQR_summary.log_minus_control_cost_mean, c=clrs[4], label="mean")
            axs.plot(range(1,n+1), LQR_summary.log_minus_control_cost_median, c="grey", label="median")
            axs.fill_between(range(1,n+1), LQR_summary.log_minus_control_cost_LB, \
                LQR_summary.log_minus_control_cost_UB,alpha=0.3, facecolor=clrs[4], label="quantiles (0.05 to 0.95)")
            axs.set_xscale('log')
            # if LQR_summary.name == 'stable_system_beta0.5_alpha2':
            #     axs.set_ylim(bottom = -200, top = 200)  
            # axs.set_yscale('log')
            axs.legend(framealpha=0.4, loc=2)

        fig.text(0.5, 0.01, 'Time', va='center', ha='center', fontsize=plt.rcParams['axes.labelsize'])
        fig.text(0.01, 0.5, 'Regret Difference', va='center', ha='center', rotation='vertical', fontsize=plt.rcParams['axes.labelsize'])

        fig.savefig("{}/Compare_Regret.pdf".format(path), bbox_inches = 'tight')


    # Regret Ratio
    cutoff = 1
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))
    with sns.axes_style("darkgrid"):
        axs.plot(range(1,n+1), LQR_summary.regret_ratio_para_mean, label = "parametric (mean)", c="blue")
        axs.plot(range(2,n+1), LQR_summary.regret_ratio_ob_mean, label = "observable (mean)", c="red")
        axs.fill_between(range(2,n+1), LQR_summary.regret_ratio_ob_LB, 
            LQR_summary.regret_ratio_ob_UB,alpha=0.2, facecolor="blue", label="quantiles (0.05 to 0.95)")
        axs.fill_between(range(1,n+1), LQR_summary.regret_ratio_para_LB, 
            LQR_summary.regret_ratio_para_UB,alpha=0.2, facecolor="red", label="quantiles (0.05 to 0.95)")
        axs.axhline(1, c="black", linewidth = 2)
        # axs.plot(range(cutoff,n), main_term_regret, c="black") # , label="main regret component"
        axs.set_xscale('log')
        # axs.set_yscale('log')
        axs.set_ylim(bottom = 0, top = 20)
        axs.set_yticks(list(axs.get_yticks()) + [1])
        axs.legend(framealpha=0.4)
        # axs.axhline(0, c="black", linewidth = 2)

    fig.text(0.5, 0.01, 'Time', va='center', ha='center', fontsize=plt.rcParams['axes.labelsize'])
    fig.text(0.01, 0.5, 'Regret Ratio', va='center', ha='center', rotation='vertical', fontsize=plt.rcParams['axes.labelsize'])

    fig.savefig("{}/Regret_ratio2.pdf".format(path), bbox_inches = 'tight')


    # Differing Convergence Rates  t \in {0, ..., t-1}
    ABK_err_L2 = LQR_summary.ABK_err_L2_mean
    AB_err_L2 = LQR_summary.AB_err_L2_mean
    K_err_L2 = LQR_summary.K_err_L2_mean # t \in {0, ..., t}
    cutoff = 100 # plot starting from here

    clrs = sns.color_palette("husl", 10)
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))
    with sns.axes_style("darkgrid"):
        axs.set_yscale('log')
        axs.set_xscale('log')
        # axs.set_ylim(bottom=0.01, top=3)
        regressor = LinearRegression()  
        # y_compensate = np.log(range(cutoff, n)) ** (LQR_summary.alpha/2) # here we adjust for the effect of alpha
        y_compensate = 1
        X_train = np.log(range(cutoff, n)).reshape(-1,1)
        y_train = np.log(K_err_L2[cutoff:n] * y_compensate) 
        fitted_reg = regressor.fit(X_train, y_train)
        axs.plot(range(cutoff, n), K_err_L2[cutoff:n], label="$\hat{K}_t - K$ slope: "+str(np.round(fitted_reg.coef_, 2)[0]), c=clrs[8])
        X_train = np.log(range(cutoff, n)).reshape(-1,1)
        y_train = np.log(AB_err_L2[cutoff:] * y_compensate) 
        axs.plot(range(cutoff, n), AB_err_L2[cutoff:], label="$[\hat{A}_t - A, \hat{B}_t - B]$ slope: "+str(np.round(fitted_reg.coef_, 2)[0]), c=clrs[4])
        X_train = np.log(range(cutoff, n)).reshape(-1,1)
        y_train = np.log(ABK_err_L2[cutoff:])
        fitted_reg = regressor.fit(X_train, y_train)
        axs.plot(range(cutoff, n), ABK_err_L2[cutoff:], label="$(\hat{A}_t - A)+(\hat{B}_t - B) \, K$ slope: "+str(np.round(fitted_reg.coef_, 2)[0]), c=clrs[0])
        axs.legend(framealpha=0.4, loc=3)   # lower left


    fig.text(0.5, 0.01, 'Time', va='center', ha='center', fontsize=plt.rcParams['axes.labelsize'])
    fig.text(0.01, 0.5, 'Error in Frobenius Norm', va='center', ha='center', rotation='vertical', fontsize=plt.rcParams['axes.labelsize'])
    fig.savefig("{}/estimation_rate.pdf".format(path), bbox_inches = 'tight')
    pass


    # Confidence Region Coverage t \in {1, 2, ..., t-1}
    clrs = sns.color_palette("husl", 10)
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))

    # print(LQR_summary.coverage_ellipsoid_K.shape)
    with sns.axes_style("darkgrid"):
        axs.axhline(0.95, c="black", linewidth = 2, label="nominal coverage (95\%)")
        axs.plot(range(1, n), LQR_summary.coverage_ellipsoid, label="$[A, B]$ confidence region", c=clrs[0])
        # the coverage only make sense when cov matrix is non-singular, which requires t >= d+m+1
        cutoff = LQR_summary.d+LQR_summary.m+1 
        # axs.plot(range(cutoff, n), LQR_summary.coverage_ellipsoid_K[cutoff-1:], label="$K$ confidence region", c=clrs[4])
        axs.plot(range(cutoff, n), LQR_summary.coverage_safety[cutoff-1:], 
            label="$\mathbb{E} [x_{t+1} \,|\, \{x_i, u_i\}_{i=0}^{t}]$ confidence region", c=clrs[8])
        axs.set_ylim(bottom=0, top = 1)
        axs.set_xscale('log')
        axs.legend(framealpha=0.4)

    fig.text(0.5, 0.01, 'Time', va='center', ha='center', fontsize=plt.rcParams['axes.labelsize'])
    fig.text(0.01, 0.5, 'Coverage', va='center', ha='center', rotation='vertical', fontsize=plt.rcParams['axes.labelsize'])
    fig.savefig("{}/Coverage.pdf".format(path), bbox_inches = 'tight')

    # Prediction Region Coverage t \in {1, 2, ..., t-1}
    clrs = sns.color_palette("husl", 10)
    fig, axs = plt.subplots(1, 1, figsize=(12, 8))
    cutoff = LQR_summary.d+LQR_summary.m+1
    with sns.axes_style("darkgrid"):
        axs.axhline(0.95, c="black", linewidth = 2, label="nominal coverage (95\%)", alpha =1)
        axs.plot(range(cutoff, n-1), LQR_summary.coverage_safety_naive[cutoff-1:], 
            label="naive prediction region", c=clrs[4], alpha =1)
        axs.plot(range(cutoff, n-1), LQR_summary.coverage_safety_our[cutoff-1:], 
            label="our prediction region", c=clrs[0], alpha =1)
        axs.set_ylim(bottom=0, top = 1)
        axs.set_xlim(left=0.6)
        axs.set_xscale('log')
        axs.legend(framealpha=0.4, loc=4)

    fig.text(0.5, 0.01, 'Time', va='center', ha='center', fontsize=plt.rcParams['axes.labelsize'])
    fig.text(0.01, 0.5, 'Coverage', va='center', ha='center', rotation='vertical', fontsize=plt.rcParams['axes.labelsize'])
    fig.savefig("{}/Safety_Coverage_True.pdf".format(path), bbox_inches = 'tight')

plot(stable_LQR_summary)
# %%
