import numpy as np
import matplotlib.pyplot as plt


class Options:
    def __init__(self):
        self.null = ''
        self.h1 = ''
        self.h2 = ''
        self.h = 0


def perform_curve_extraction(M, t='', options='', leng=10):

    # perform_curve_extraction - extract level set curves from an image.
    #
    #   c_list = perform_curve_extraction(M, t, options)
    #
    #   'M' is the image.
    #   't' is the level.
    #   'c_list' is a cell array of 2D curves.
    #   'options' is an optional structure with fields: 
    #       'max_nb' is the number of curves extracted (only the 'nb' longest).
    #       'min_length' is the minimum length (in *pixels*) of extracted curves.
    #
    #   The image is always assumed to lie in [0,1]x[0,(ny-1)/(nx-1)]
    #   where [nx,ny] = size(M).
    #
    #   Copyright (c) 2004 Gabriel Peyré

    if t == '':
        t = 0
    if options == '':
        options = Options()
    
    [nx, ny] = M.shape
    #x = 0:1/(nx-1):1
    x = [i / (nx - 1) for i in range(nx)]
    #y = ( 0:1 / (ny - 1):1 ) * ((ny - 1) / (nx - 1))
    y = [1 / (nx - 1) * i  for i in range(ny)]
    #c = contourc(x, y, M.T, [t, t])
    c = plt.contour(x, y, M.T, [0])
    #del c_list
    c_list = list()

    for path in c.collections[0].get_paths():
        c_list.append(path.vertices)

    n = len(c_list)
    c_list2 = list()

    # filter by nb
    if options == 'max_nb':
        max_nb = leng
        l = np.zeros((n, 1))
        for i in range(n):
            l[i] = len(c_list[i])

        [tmp, I] = sort(l)
        I = reverse(I)
        I = I[ 1:[min(max_b, n)] ]
        c_list1 = c_list
        c_list = list()
        for i in range(I):
            c_list[i] = c_list1[I(i)];       

    # filter by size
    if options == 'min_length':
        min_length = leng
        '''c_list1 = c_list
        c_list = list()
        k = 0'''
        for i in range(n):
            l = len(c_list[i])
            if l >= min_length:
                c_list2.append(c_list[i])

    return c_list2
