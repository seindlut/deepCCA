""" Linear algebra tools """
from cca_linear import cca as CCA
import numpy as np
from numpy.linalg import inv, lstsq, cholesky
from scipy.linalg import sqrtm

def order_cost(H1,H2):
    ''' Run linear CCA and return correlation'''
    a,b,x,y=CCA(H1,H2)
    return cor_cost(x,y), cca(x,y)

def mat_pow(matrix):
    return sqrtm(inv(matrix))

def stable_inverse_Aneg1_dot_B(A,B):
    return lstsq(A,B)[0]

def solve_cholesky(A,B):
    L=cholesky(A)
    #assert np.allclose(np.dot(L, L.T),A)
    y=lstsq(L,B)[0]
    return lstsq(L.T,y)

def stable_inverse_Aneg1_dot_B_cholesky(A,B):
    # solves x=A^-1.B or A.x=Bm where A=L.L.T
    ##L=cholesky(A)
    ##assert np.allclose(np.dot(L, L.T),A)
    ##y=lstsq(L,B)[0]
    return solve_cholesky(A,B)[0]
    ##return lstsq(L.T,y)[0]

def stable_inverse_A_dot_Bneg1(A,B):
    # solves x=A.B^-1 or x.A=B
    return lstsq(B, A.T)[0].T

def stable_inverse_A_dot_Bneg1_cholesky(A,B):
    # solves x=A.B^-1 or x.A=B
    return solve_cholesky(B, A.T)[0].T

def mat_pow2(matrix):
    return inv(sqrtm(matrix))

def cor_cost(H1,H2):
    cor=0.0
    for i in range(H1.shape[1]):
        cur_cor = abs(np.corrcoef(H1[:,i], H2[:,i])[0,1])
        if not np.isnan(cur_cor):
            cor += cur_cor
        else:
            cor += 1.0
    return cor

def cca_cost(H1, H2):
    return (cca(H1, H2)+cca(H2, H1))/(cca(H1, H1)+cca(H2, H2))

def cca(H1, H2):
    H1bar = copy.deepcopy(H1)
    H1bar = H1bar-H1bar.mean(axis=0)
    H2bar = copy.deepcopy(H2)
    H2bar = H2bar-H2bar.mean(axis=0)
    H1bar = H1bar.T
    H2bar = H2bar.T
    H1bar += np.random.random(H1bar.shape)*0.00001
    H2bar += np.random.random(H2bar.shape)*0.00001
    r1 = 0.00000001
    m = H1.shape[0]

    SigmaHat12 = (1.0/(m-1))*np.dot(H1bar, H2bar.T)
    SigmaHat11 = (1.0/(m-1))*np.dot(H1bar, H1bar.T)
    SigmaHat11 = SigmaHat11 + r1*np.identity(SigmaHat11.shape[0], dtype=np.float32)
    SigmaHat22 = (1.0/(m-1))*np.dot(H2bar, H2bar.T)
    SigmaHat22 = SigmaHat22 + r1*np.identity(SigmaHat22.shape[0], dtype=np.float32)
    SigmaHat11_2=mat_pow(SigmaHat11).real.astype(np.float32)
    SigmaHat22_2=mat_pow(SigmaHat22).real.astype(np.float32)

    TMP2 = stable_inverse_A_dot_Bneg1(SigmaHat12, sqrtm(SigmaHat22))#np.dot(SigmaHat12, SigmaHat22_2)
    TMP3 = stable_inverse_A_dot_Bneg1_cholesky(SigmaHat12, sqrtm(SigmaHat22))#np.dot(SigmaHat12, SigmaHat22_2)

    Tval = stable_inverse_Aneg1_dot_B(sqrtm(SigmaHat11), TMP2)
    Tval3 = stable_inverse_Aneg1_dot_B_cholesky(sqrtm(SigmaHat11), TMP3)

    corr =  np.trace(sqrtm(np.dot(Tval.T, Tval)))
    return corr

def cca_prime(H1, H2):
    H1bar = copy.deepcopy(H1)
    H1bar = H1bar-H1bar.mean(axis=0)
    H2bar = copy.deepcopy(H2)
    H2bar = H2bar-H2bar.mean(axis=0)
    H1bar = H1bar.T
    H2bar = H2bar.T
    H1bar += np.random.random(H1bar.shape)*0.00001
    H2bar += np.random.random(H2bar.shape)*0.00001
    r1 = 0.00000001
    m = H1bar.shape[0]
    SigmaHat12 = (1.0/(m-1))*np.dot(H1bar, H2bar.T)
    SigmaHat11 = (1.0/(m-1))*np.dot(H1bar, H1bar.T)
    SigmaHat11 = SigmaHat11 + r1*np.identity(SigmaHat11.shape[0], dtype=np.float32)
    SigmaHat22 = (1.0/(m-1))*np.dot(H2bar, H2bar.T)
    SigmaHat22 = SigmaHat22 + r1*np.identity(SigmaHat22.shape[0], dtype=np.float32)
    SigmaHat11_2=mat_pow(SigmaHat11).real.astype(np.float32)
    SigmaHat22_2=mat_pow(SigmaHat22).real.astype(np.float32)
    TMP3 = stable_inverse_A_dot_Bneg1_cholesky(SigmaHat12, sqrtm(SigmaHat22))
    Tval = stable_inverse_Aneg1_dot_B_cholesky(sqrtm(SigmaHat11), TMP3)
    U, D, V, = np.linalg.svd(Tval)
    D=np.diag(D)
    UVT = np.dot(U, V)
    Delta12 = np.dot(SigmaHat11_2, np.dot(UVT, SigmaHat22_2))
    UDUT = np.dot(U, np.dot(D, U.T))
    Delta11 = (-0.5) * np.dot(SigmaHat11_2, np.dot(UDUT, SigmaHat11_2))
    grad_E_to_o = (1.0/m) * (2*np.dot(Delta11,H1bar)+np.dot(Delta12,H2bar))

    return -1.0*grad_E_to_o.T##np.real(grad_E_to_o.real).T
