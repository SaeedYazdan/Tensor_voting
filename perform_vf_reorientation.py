import numpy as np


def perform_vf_reorientation(v, method='', w=''):

    # perform_vf_reorientation - try to reorient the vf.
    #
    #   w = perform_vf_reorientation(v, method);
    #
    #   'method' can be 'xproj', 'yproj', 'circproj' or 'localproj' or
    #   'custproj'.
    #
    #   For 'custproj' you have to provide an additional vector w.
    #
    #   Copyright (c) 2004 Gabriel Peyré

    if method == '':
        method = 'xproj'

    if method.lower() == 'localproj':
        # special case.
        #n = size(v, 1)
        #p = size(v, 2)
        [n, p] = v.shape
        for i in range(n):
            for j in range(n):
                m = zeros(1, 1, 2)
                if i > 0:
                    m = m + v[i-1, j, :]
                
                if j > 0:
                    m = m + v[i, j-1, :]
                
                if i > 0 and j > 0:
                    m = m + v[i-1, j-1, :]
                
                s = np.dot( m, v[i, j, :] )
                if s > 0:
                    v[i, j, :] = v[i, j, :] * np.sign(s)
                   
    if method.lower() == 'xproj':   
        s = v[:, :, 0]
    elif method.lower() == 'yproj':
        s = v[:, :, 1]
    elif method.lower() == 'circproj':
        [n, p] = v.shape
        [Y, X] = np.meshgrid(p, n)
        s = v[:, :, 0] * X + v[:, :, 1] * Y
    elif method.lower() == 'custproj':
        s = v[:, :, 0] * w[0] + v[:, :, 1] * w[1]
    else:
        raise Exception('Unknown method')

    w = v
    w[:, :, 0] = v[:, :, 0] * np.sign(s)
    w[:, :, 1] = v[:, :, 1] * np.sign(s)

    return w
        
        
