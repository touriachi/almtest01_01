import json
import logging
import logging.config
import os
from os import error

import jsonschema

import copy
import time
import numpy as np
import pandas as pd
import math
import shelve
import dill
from   scipy  import  array, linalg, dot

from pandas import ExcelWriter
from pandas import ExcelFile

def get_normd_matrix(x,matrix,years):
    tt = np.apply_along_axis(lambda x : np.random.normal(0, 1, 5000), 0, matrix)

    # print(tt.mean(axis=0))
    # print(tt.std(axis=0))


    df = pd.DataFrame(tt,columns=years)
    df.index = x.index
    x= x.drop(columns=years)
    x = x.join(df)

    # print(x['Year1'].mean(axis=0))
    # print(x['Year1'].std(axis=0))
    return x


def check_matrix(x, years):

    print(x['Year1'].mean(axis=0))
    print(x['Year1'].std(axis=0))
    return x



def nearPD (x, corr = True, keepDiag = False, do2eigen = True ,
          doSym = False, doDykstra = True, only_values = False,
          eig_tol = 1e-06, conv_tol = 1e-07, posd_tol = 1e-08, maxit = 200,
          conv_norm_type = "fro", trace = False) :

    if  np.any(np.abs(x - x.T)) > conv_tol  :
        print ("Matrix no sysmetric")
        return False

    n = x.shape[1]
    if (keepDiag) :
        diagX0 = np.diag(x)
    if (doDykstra) :
        D_S = x.copy()
        D_S[:,:] = 0

    X = x.copy()
    iter = 0
    converged = False
    conv =math.inf

    while (iter < maxit and not converged) :
        Y = X.copy()
        if (doDykstra) :
            R =  Y - D_S
            eigenValues, eigenVectors =np.linalg.eig(R)
        else :
            eigvals, eigvecs = np.linalg.eigh(Y)

        idx = eigenValues.real.argsort()[::-1]
        d = eigenValues.real[idx]
        Q = eigenVectors[:, idx]


        p = [v > eig_tol*d[0] for v in d]
        if (not any(p)) :
             print("Matrix seems negative semi-definite")

        pp = [i for i, x in enumerate(p) if not x]
        Q = np.delete(Q, pp, axis=1)

        tt= array([d[p],] * Q.shape[0])
        tt2= np.multiply(Q,tt)
        X = np.dot(tt2, Q.transpose())


        if doDykstra :
            D_S = X - R
        if doSym:
            X = (X + X.transpose()) / 2
        if corr:
            np.fill_diagonal(X,1)
        else :
            if keepDiag :
                 np.fill_diagonal(X,diagX0)

        conv =np.linalg.norm(Y-X, np.inf) / np.linalg.norm(Y, np.inf)
        iter = iter + 1
        converged = conv <= conv_tol

    if not converged :
        print ("not converged")
    else:
        print  ("converged")
        print (iter)

    if (do2eigen or  only_values) :
        eigenValues, eigenVectors = np.linalg.eigh(X)
        idx = eigenValues.real.argsort()[::-1]
        d = eigenValues.real[idx]
        eigvecs = eigenVectors[:, idx]

        Eps = posd_tol * abs(d[0])
        if d[n-1] < Eps :
            d[d < Eps] = Eps
            if not only_values :
                Q  = eigvecs
                o_diag = X.diagonal()
                pp=(Q * d).T
                X= np.dot(Q, pp)
                D  = np.sqrt(np.maximum(Eps, o_diag) / X.diagonal())
                tt=array([D, ] * n)
                X = D * X * tt
        if only_values :
            return (d)
        if corr :
             np.fill_diagonal(X,1)
        else :
            if keepDiag :
             np.fill_diagonal(X, diagX0)
    return  X



class MatrixOperations:

    def near_pd(self, A, nit=10):
        n = A.shape[0]
        W = np.identity(n)
        # W is the matrix used for the norm (assumed to be Identity matrix here)
        # the algorithm should work for any diagonal W
        deltaS = 0
        Yk = A.copy()
        for k in range(nit):
            Rk = Yk - deltaS
            Xk = self.__get_ps(Rk, W=W)
            deltaS = Xk - Rk
            Yk = self.__get_pu(Xk, W=W)
        return Yk

    def __get_aplus(self, A):
        eigval, eigvec = np.linalg.eig(A)
        Q = np.mat(eigvec)
        xdiag = np.mat(np.diag(np.maximum(eigval, 0)))
        return Q * xdiag * Q.T

    def __get_ps(self, A, W=None):
        W05 = np.mat(W ** .5)
        return W05.I * self.__get_aplus(W05 * A * W05) * W05.I

    def __get_pu(self, A, W=None):
        Aret = np.array(A.copy())
        Aret[W > 0] = np.array(W)[W > 0]
        return np.mat(Aret)


def check_symmetric(matrix, tol=1e-5):
    return not False in (np.abs(matrix-matrix.T) < tol)


def is_positive_semi_definite(matrix,logger,epsilon=1e-07):

    #is symetric matrix
    if not check_symmetric(matrix):
        logger.debug("Correlation matrix is not symetric ")
        return False
    logger.debug("Correlation matrix is symetric ")

    # find eign values and check if they are positive  ( same approach a R language
    if  np.all(np.linalg.eigvals(matrix) > -epsilon) :
        logger.debug("Following eign values approach, the correlation matrix is postive semi-definite")
        return True
    else :
        logger.debug("Following eign values approach, the  correlation matrix is not positive semi-definite")
        return False


def get_previous_colunmn ( cols , col_name) :
   try :
      pos= cols.get_loc(col_name)
      pos = pos - 1
   except :
      pos =-1

   if pos >= 0 :
      colname = cols[pos]
      return colname
   else :
      return -1


def save_workspace(var_value,var_name,filename='shelve.out'):
    my_shelf = shelve.open(filename, 'n')  # 'n' for new
    my_shelf[var_name]=var_value
    my_shelf.close()

def get_workspace(var_name,filename='shelve.out'):
    my_shelf = shelve.open(filename)
    v = my_shelf[var_name]
    my_shelf.close()
    return v