import numpy as np

import compute_grad as cg



class Options:
    def __init__(self):
        self.null = 0
        self.h1 = 0
        self.h2 = 0
        self.h = 0


def compute_rigidity_tensor(M, options=''):

    # compute_rigidity_tensor - compute the rigidity
    #   tensorial field, which is
    #       [(dM/dx)^2    dM/dx*dM/dy]
    #       [dM/dx*dM/dy    (dM/dy)^2]
    #
    #   H = compute_rigidity_tensor(M,options);
    #
    #   Copyright (c) 2004 Gabriel Peyré

    if options == '':
        options = Options()

    if options.h1 == 0:
        options.h1 = 1
    h1 = options.h1
    if options.h2 == 0:
        options.h2 = 1
    h2 = options.h2

    if options.h == 0:
        h1 = options.h
        h2 = options.h

    [n, p] = M.shape

    grad = cg.compute_grad(M, options)
    dx = grad[:, :, 0]
    dy = grad[:, :, 1]

    H = np.zeros((n, p, 2, 2))
    H[:, :, 0, 0] = dx * dx
    H[:, :, 1, 1] = dy * dy
    H[:, :, 0, 1] = dx * dy
    H[:, :, 1, 0] = H[:, :, 0, 1]
    return H
