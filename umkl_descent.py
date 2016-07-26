# The function umkl_descent implements a coordinate descent
# algorithm for unsupervised multiple-kernel learning.
# The main script should be run with a .npy file as the
# command line argument. The .npy file should contain the
# multiple kernels as a collection of sparse scipy matrices.

from __future__ import division
import numpy as np
from scipy.linalg import *
import sklearn
import sys

norm = np.linalg.norm

def umkl_descent(kernels, rho, epsilon=0.001, p=10, sigma=None):
    # Obtain k_i from eigenvalue decompositions 
    # of given kernels. (Only p largest eigenvalues)
    n = kernels[0].shape[0]
    print 'Peforming UMKL for ' + str(n) + ' X ' + str(n) + ' kernels.'
    q = kernels[0].shape[1]
    w, K = eigh(kernels[0], eigvals=(q-p,q-1))
    for i in range(K.shape[1]):
        K[:,i] *= np.sqrt(w[i])
    for kernel in kernels[1:]:
        w, v = eigh(kernel, eigvals=(q-p,q-1))
        for i in range(v.shape[1]):
            v[:,i] *= np.sqrt(w[i])
        K = np.hstack((K, v))

    # Normalize K matrix
    m = K.shape[1]
    k_norms = [norm(K[:,i]) for i in range(m)]
    K /= sum(k_norms)
    
    # Eliminate k_i with norm < rho
    to_delete = []
    for i in range(m):
        if norm(K[:,i]) < rho:
            to_delete.append(i)
    K = np.delete(K, to_delete, 1)
    m = K.shape[1]
    if m == 0:
        print "Rho too large."
        exit(0)
    k_norms = [norm(K[:,i]) for i in range(m)]

    # Initialize U and compute initial objective value
    U = np.random.randn(m, n)
    obj_term1 = rho*sum([norm(U[i,:]) for i in range(m)])

    Z = np.eye(n)
    Id = np.eye(n)

    if sigma is not None:
        Z = np.vstack((Z, np.zeros((m, n))))
        K = np.vstack((K, sigma*np.eye(m)))
        Id = np.eye(m+n)
    
    for i in range(m):
        Z -= np.outer(K[:,i], U[i,:])
    
    prev_obj = obj_term1 + norm(Z)

    # Descent loop
    objective_values = [prev_obj]
    diff = prev_obj
    while diff > epsilon:
        for i in range(m):
            # Update Z 
            Z += np.outer(K[:,i], U[i,:])

            # Actual descent
            Z_norm = norm(Z)
            kTZ_norm  = norm(np.dot(K[:,i].T, Z))

            alpha_0 = 1.0 / (k_norms[i]**2)
            c = k_norms[i]
            d = alpha_0 * ( Z_norm**2 / kTZ_norm**2 - alpha_0 )
            alpha = alpha_0 - np.sqrt( (rho**2 * d) / (c**2 - rho**2) )

            f_of_alpha = rho*abs(alpha) * kTZ_norm + norm( np.dot(Id - alpha*np.outer(K[:,i], K[:,i].T), Z) )
            if Z_norm < f_of_alpha:
                alpha = 0

            if np.isnan(alpha):
                print "Alpha imaginary."
                exit(0)
            if alpha < 0:
                print "Alpha negative."
                exit(0)

            # Descend and update objective value
            temp = obj_term1 - rho*norm(U[i,:])
            U[i,:] = alpha * np.dot(K[:,i].T,  Z)
            temp += rho*norm(U[i,:])
            obj_term1 = temp

            Z -= np.outer(K[:,i], U[i,:])
            new_obj = obj_term1 + norm(Z)
            objective_values.append(new_obj)

        # Check convergence
        diff = prev_obj - new_obj
        prev_obj = new_obj

        if diff < 0:
            print "Objective value increased."
            exit(0)

    # Recover primal variables: optimal weights
    Phi = new_obj
    print "Optimal value:", Phi
    weights = [ norm(Z)/Phi ]

    for i in range(m):
        weights.append( rho*norm(U[i,:])/Phi )

    # Compute optimal kernel
    #optimal_kernel = weights[0]*np.eye(n)
    #for i in range(m):
    #    optimal_kernel += (rho**(-2)) * weights[i] * np.outer(K[:,i], K[:,i])
    
    return weights, objective_values
    

if __name__ == '__main__':
    kernels_file = sys.argv[1]
    kernels = np.load(kernels_file)
    kernels = [k.todense() for k in kernels]
    r = 0.01
    s = r/20.0
    weights, objective_values = umkl_descent(kernels, rho=r, epsilon=1e-6, sigma=s)

