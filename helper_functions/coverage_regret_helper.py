import numpy as np
from control import dlyap
# from fc_helper import solve_F_finite, solve_P_finite
from numpy.linalg import inv

def solve_P_finite(A, B, R, Q, N, step=100):
    from control import dare
    P = dare(A, B, Q, R)
    return P[0]


def solve_F_finite(A, B, R, Q, N, step=100):
    from control import dare
    F = dare(A, B, Q, R)
    return np.array(F[-1])

# return the regret gradient
def get_regret_grad(A, B, K, P):
    d = A.shape[0]
    m = B.shape[1]
    L = A + B.dot(K)
    regret_grad = np.zeros(d*(d+m))
    # regret_grad_brute = np.zeros(d*(d+m))
    for i in range(d):
        for j in range(d+m):
        # set the dA dB
            dA = np.zeros([d, d])
            dB = np.zeros([d, m])
            if j < d:
                dA[i, j] = 1
            else:
                dB[i, j-d] = 1

            # solve dP/dA, dB
            # expected shape (d,d,d,d+m), the first two dim for dP, the other for dA,dB
            # Lt dP L - dP + (dA + dB K)t P L + Lt P (dA + dB K)
            dAdBK = dA + dB.dot(K)
            tmpQ = dAdBK.T.dot(P).dot(L)
            Q_lyap = tmpQ + tmpQ.T
            tmp_L = np.copy(L.T)
            dP = dlyap(A = tmp_L, Q = Q_lyap)

            # regret gradient
            # trace(dB.T P B + B.T dP B + B.T P dB)
            dBT_P_B = dB.T.dot(P).dot(B)
            BT_dP_B = B.T.dot(dP).dot(B)
            d_regret = np.trace(dBT_P_B + dBT_P_B.T + BT_dP_B)

            index = i*(d+m) + j
            regret_grad[index] =  d_regret

            # sanity check
            # step_size = 0.01
            # P_h = solve_P_finite(A + dA*step_size, B + dB*step_size, R, Q, N)
            # dP_brute = (P_h-P)/step_size

            # dBT_P_B = dB.T.dot(P).dot(B)
            # BT_dP_B = B.T.dot(dP_brute).dot(B)
            # d_regret_brute = dBT_P_B + dBT_P_B.T + BT_dP_B
            # regret_grad_brute[index] =  d_regret_brute

    return(regret_grad)



def get_P_grad(A, B, K, P):
    d = A.shape[0]
    m = B.shape[1]
    L = A + B.dot(K)
    P_grad = np.zeros([d, d, d*(d+m)])
    # regret_grad_brute = np.zeros(d*(d+m))
    for i in range(d):
        for j in range(d+m):
        # set the dA dB
            dA = np.zeros([d, d])
            dB = np.zeros([d, m])
            if j < d:
                dA[i, j] = 1
            else:
                dB[i, j-d] = 1

            # solve dP/dA, dB
            # expected shape (d,d,d,d+m), the first two dim for dP, the other for dA,dB
            # Lt dP L - dP + (dA + dB K)t P L + Lt P (dA + dB K)
            dAdBK = dA + dB.dot(K)
            tmpQ = dAdBK.T.dot(P).dot(L)
            Q_lyap = tmpQ + tmpQ.T
            tmp_L = np.copy(L.T)
            dP = dlyap(A = tmp_L, Q = Q_lyap)

            index = i*(d+m) + j
            P_grad[:, :, index] =  dP

    return(P_grad)

def mean_UB_LB_std(nparray):
    nparray_mean = np.mean(np.sum(np.abs(nparray), axis=2), axis=0) # shape n
    nparray_L1_UB = np.quantile(np.sum(np.abs(nparray), axis=2), 0.95, axis=0)
    nparray_L1_LB = np.quantile(np.sum(np.abs(nparray), axis=2), 0.05, axis=0)
    nparray_std = np.std(nparray, axis=0)
    result_dict = {
        "mean": nparray_mean,
        "UB": nparray_L1_UB,
        "LB": nparray_L1_LB,
        "std": nparray_std
    }
    return(result_dict)

def get_K_grad(A, B, K, P, R):
    d = A.shape[0]
    m = B.shape[1]
    L = A + B.dot(K)
    P_grad = get_P_grad(A, B, K, P) # shape d, d, d*(d+m)
    K_grad = np.zeros([m, d, d*(d+m)])
    # print(K_grad.shape)
    # regret_grad_brute = np.zeros(d*(d+m))
    for i in range(d):
        for j in range(d+m):
        # set the dA dB
            dA = np.zeros([d, d])
            dB = np.zeros([d, m])
            index = i*(d+m) + j
            dP = P_grad[:,:, index] # shape d, d
            if j < d:
                dA[i, j] = 1
            else:
                dB[i, j-d] = 1

            # solve dK/dA, dB
            # expected shape (d,d,d,d+m), the first two dim for dP, the other for dA,dB
            # Lt dP L - dP + (dA + dB K)t P L + Lt P (dA + dB K)
            R_plus_BTPB_inv = inv(R + B.T.dot(P).dot(B))
            dBTPL = dB.T.dot(P).dot(L)
            dL = dA + dB.dot(K) # define dL
            BTPdL = B.T.dot(P).dot(dL)
            BTdPL = B.T.dot(dP).dot(L)
            dK = -R_plus_BTPB_inv.dot(dBTPL + BTPdL + BTdPL)
            # print(dK)
            # print(index)

            K_grad[:, :, index] =  dK
    K_grad = K_grad.reshape(-1, d*(d+m))

    return(K_grad)

# return three terms with shape m*d, d, (d+m)
# K_grad_dBTPL, K_grad_BTPdL, K_grad_BTdPL
def get_K_grad_three_parts(A, B, K, P, R):
    d = A.shape[0]
    m = B.shape[1]
    L = A + B.dot(K)
    P_grad = get_P_grad(A, B, K, P) # shape d, d, d*(d+m)
    K_grad_dBTPL = np.zeros([m, d, d*(d+m)])
    K_grad_BTPdL = np.zeros([m, d, d*(d+m)])
    K_grad_BTdPL = np.zeros([m, d, d*(d+m)])
    # print(K_grad.shape)
    # regret_grad_brute = np.zeros(d*(d+m))
    for i in range(d):
        for j in range(d+m):
        # set the dA dB
            dA = np.zeros([d, d])
            dB = np.zeros([d, m])
            index = i*(d+m) + j
            dP = P_grad[:,:, index] # shape d, d
            if j < d:
                dA[i, j] = 1
            else:
                dB[i, j-d] = 1

            # solve dK/dA, dB
            # expected shape (d,d,d,d+m), the first two dim for dP, the other for dA,dB
            # Lt dP L - dP + (dA + dB K)t P L + Lt P (dA + dB K)
            R_plus_BTPB_inv = inv(R + B.T.dot(P).dot(B))
            dBTPL = dB.T.dot(P).dot(L)
            dL = dA + dB.dot(K) # define dL
            BTPdL = B.T.dot(P).dot(dL)
            BTdPL = B.T.dot(dP).dot(L)
            dK_dBTPL = -R_plus_BTPB_inv.dot(dBTPL)
            dK_BTPdL = -R_plus_BTPB_inv.dot(BTPdL)
            dK_BTdPL = -R_plus_BTPB_inv.dot(BTdPL)
            # print(dK)
            # print(index)

            K_grad_dBTPL[:, :, index] =  dK_dBTPL
            K_grad_BTPdL[:, :, index] =  dK_BTPdL
            K_grad_BTdPL[:, :, index] =  dK_BTdPL
    K_grad_dBTPL = K_grad_dBTPL.reshape((m*d, d, (d+m)))
    K_grad_BTPdL = K_grad_BTPdL.reshape((m*d, d, (d+m)))
    K_grad_BTdPL = K_grad_BTdPL.reshape((m*d, d, (d+m)))
    three_sum =  K_grad_dBTPL + K_grad_BTPdL + K_grad_BTdPL

    return(K_grad_dBTPL, K_grad_BTPdL, K_grad_BTdPL, three_sum)

