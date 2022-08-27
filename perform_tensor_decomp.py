import numpy as np
import math
import prod_vf_sf as pvs


def perform_tensor_decomp(T, order=''):

    # perform_tensor_decomp - perform an eigendecomposition.
    #
    #   [e1,e2,l1,l2] = perform_tensor_decomp(T);
    #
    #   'e1(i,j,:)' is the main eigenvector at location (i,j)
    #       with associated largest eigenvalue 'l1(i,j)'.
    #   'e2(i,j,:)' is the second eigenvector at location (i,j)
    #       with associated smallest eigenvalue 'l2(i,j)'.
    #
    #   So you always have l1>=l2 (not in absolute value !).
    #
    #   If 'order'=='abs' then the the decomposition is done
    #   so that abs(l1)>=abs(l2)
    #
    #   'T' must be a tensorial field (produced eg. by compute_hessian), 
    #   so it should be symmetric.
    #
    #   Copyright (c) 2004 Gabriel Peyré

    # retrieve the 4 entries of the tensor field
    K11 = T[:, :, 0, 0]
    K12 = T[:, :, 0, 1]
    K21 = T[:, :, 1, 0]
    K22 = T[:, :, 1, 1]

    [n, p] = K11.shape

    e1 = np.zeros((n, p, 2))
    e2 = np.zeros((n, p, 2))
    l1 = np.zeros((n, p))
    l2 = np.zeros((n, p))

    # trace/2
    t = (K11 + K22) / 2

    a = K11 - t
    b = K12

    # ab2 = math.sqrt(a ** 2 + b ** 2)
    ab2 = np.zeros((n, n))
    theta = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ab2[i, j] = math.sqrt(a[i, j] ** 2 + b[i, j] ** 2)
            theta = math.atan2(ab2[i, j] - a[i, j], b[i, j])
            e1[i, j, 0] = math.cos(theta)
            e1[i, j, 1] = math.sin(theta)
            e2[i, j, 0] = -math.sin(theta)
            e2[i, j, 1] = math.cos(theta)
            
            
    l1 = ab2 + t
    l2 = - ab2 + t

    #theta = atan2( ab2 - a, b )

    #e1[:, :, 1] = math.cos(theta)
    #e1[:, :, 2] = math.sin(theta)
    #e2[:, :, 1] = -math.sin(theta)
    #e2[:, :, 2] = math.cos(theta)

    if order == 'abs':
        # reorder the eigenvalue according to absolute value.
        A = abs(l1) > abs(l2)
        
        ee1 = pvs.prod_vf_sf(e1, A+1-1) + pvs.prod_vf_sf(e2, 1 - A)
        e2 = pvs.prod_vf_sf(e1, 1 - A) + pvs.prod_vf_sf(e2, A+1-1)
        e1 = ee1
        ll1 = l1 * A + l2 * (1 - A)
        l2 = l1 * (1 - A) + l2 * A
        l1 = ll1

    return [e1, e2, l1, l2]
