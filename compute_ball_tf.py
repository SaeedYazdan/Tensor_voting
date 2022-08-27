import numpy as np
import math

import compute_stick_tf as cst



def compute_ball_tf(N='', n='', sigma='', c='', p=''):

    # compute_ball_tf - compute the voting kernel of a ball tensor.
    #
    #   T = compute_ball_tf(N,n,sigma,c,p);
    #
    #   'N' is the total size of the image (supposed to lie in [0,1]x[0,1]).
    #
    #       The equation for the decay of the field is
    #           DF(s,k) = exp(-(s^2+c*k^2)/sigma^2)
    #       where 's' is the length of the circle that joins
    #       the two points, and 'k' is its curvature.
    #
    #   Optional:
    #   'sigma' control the scale of the voting field.
    #   'c' control the ratio between distance and curvature.
    #       The higher 'c' is, the narrower the field will be.
    #   'n' is the size of the kernel (should be odd).
    #
    #   'p' is the number of orientation samples needed to compute the field.
    #       (the higher the slower the computation).
    #
    #   See also compute_stick_tf.
    #
    #   Copyright (c) 2004 Gabriel Peyré


    if N == '':
        raise Exception('Not enough arguments');
    if n == '':
        n = N / 2
    if sigma == '':
        sigma = 0.2
    if c == '':
        c = 0.1 * sigma
    if p == '':
        p = 16

    n = int(n / 2) * 2 + 1
    nn = (n - 1) / 2

    T = np.zeros((n, n, 2, 2))
    print('p', p)
    #for theta in range ((0, 1/p, 1-1/p)*2*math.pi):
    for i in range(p):

        theta = (i / p) * 2 * math.pi
    
        v = [math.cos(theta), math.sin(theta)]
        B = cst.compute_stick_tf(v, N, n, sigma, c)
        T = T + B
    T = T / p
    return T
