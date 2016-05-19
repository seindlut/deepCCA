from __future__ import division
from numpy.linalg import lstsq,eig
from numpy import cov,dot,arange,c_
import numpy as np

def cca(x_tn,y_tm, reg=0.00000001):
    # Centering
    x_tn = x_tn-x_tn.mean(axis=0)
    y_tm = y_tm-y_tm.mean(axis=0)
    # Space dimension
    N = x_tn.shape[1]
    M = y_tm.shape[1]
    # Concatenate and evluate the empiric covar matrix
    xy_tq = c_[x_tn,y_tm]
    cqq = cov(xy_tq,rowvar=0)
    # Covar matrices with regularization
    cxx = cqq[:N,:N]+reg*np.eye(N)+0.000000001*np.ones((N,N))
    cxy = cqq[:N,N:(N+M)]+0.000000001*np.ones((N,M))
    cyx = cqq[N:(N+M),:N]+0.000000001*np.ones((M,N))
    cyy = cqq[N:(N+M),N:(N+M)]+reg*np.eye(M)+0.000000001*np.ones((M,M))

    K = min(N,M)

    xldivy = lstsq(cxx,cxy)[0]
    yldivx = lstsq(cyy,cyx)[0]

    _,vecs = eig(dot(xldivy,yldivx))
    a_nk = vecs[:,:K]
    b_mk = dot(yldivx,a_nk)

    u_tk = dot(x_tn,a_nk)
    v_tk = dot(y_tm,b_mk)

    return a_nk,b_mk,u_tk,v_tk

def normr(a):
    return a/np.sqrt((a**2).sum(axis=1))[:,None]

def test_cca():
    # inputs :
    x_tn = 1/np.arange(1,31).reshape(6,5)
    y_tm = 1/np.arange(1,19).reshape(6,3)

    a,b,u,v = cca(x_tn,y_tm)
    print "Mappings : ", a.shape, b.shape
    print 'Output covariance: ',np.sum(cov(u,v))
    print 'Input covariance: ', np.sum(cov(x_tn.T,y_tm.T))

if __name__ == "__main__":
    test_cca()
