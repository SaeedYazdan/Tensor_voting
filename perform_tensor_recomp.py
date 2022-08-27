import numpy as np



def perform_tensor_recomp(e1, e2, l1, l2):

    # perform_tensor_recomp - create the tensor field corresponding to the given eigendecomposition.
    #
    #   T = perform_tensor_recomp(e1,e2,l1,l2);
    #
    #   'e1(i,j,:)' is the main eigenvector at location (i,j)
    #       with associated largest eigenvalue 'l1(i,j)'.
    #   'e2(i,j,:)' is the second eigenvector at location (i,j)
    #       with associated smallest eigenvalue 'l2(i,j)'.
    #
    #   You have 
    #       T = l1*e1*e1' + l2*e2*e2'
    #
    #   Copyright (c) 2004 Gabriel Peyré


    T = np.zeros((l1.shape[0], l1.shape[1], 2, 2))
    T[:, :, 0, 0] = l1 * e1[:, :, 0] ** 2 + l2 * e2[:, :, 0] ** 2
    T[:, :, 0, 1] = l1 * e1[:, :, 0] * e1[:, : ,1] + l2 * e2[:, :, 0] * e2[:, :, 1]
    T[:, :, 1, 0] = T[:, :, 0, 1]
    T[:, :, 1, 1] = l1 * e1[:, :, 1] ** 2 + l2 * e2[:, :, 1] ** 2

    return T
