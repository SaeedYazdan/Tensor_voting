import numpy as np



def plot_vf(plt, vf, M='', is_oriented='', reorient=False):

    # plot_vf - plot a vector field with 
    #   an optional image in the background.
    #
    # plot_vf(vf, M);
    #
    #   Copyright (c) 2004 Gabriel Peyré

    if M == '':
        M = []
    if is_oriented == '':
        is_oriented = 1
    if reorient == '':
        reorient = 0

    if reorient:
        # reorient the vf to x>0
        epsi = np.sign(vf[:, :, 1])
        I = find(epsi == 0)
        epsi[I] = 1
        vf[:, :, 1] = vf[:, :, 1] * epsi
        vf[:, :, 2] = vf[:, :, 2] * epsi

    [n, p] = vf.shape[:2]

    #x = 0:1/(n-1):1
    x = [i/(n-1) for i in range(n)]
    #y = 0:1/(p-1):1
    y = [i/(p-1) for i in range(p)]
    [Y, X] = np.meshgrid(y, x)
    #x=x*M.shape[0]
    #y=y*M.shape[1]
    #print('x', x)
    #print('y', y)
    #print('M', M.shape)
    #breakpoint()
    
     
    #hold on
    plt.imshow(M.T, origin='lower')
    if is_oriented:
        plt.quiver(X * (M.shape[0] - 1), Y * (M.shape[1] - 1), vf[:, :, 0], vf[:, :, 1], scale_units='xy', scale = 1, color='r')
        pass
    else:
        plt.quiver(X, Y, vf[:, :, 0], vf[:, :, 1], 0.4, color='r')
        plt.quiver(X, Y, -vf[:, :, 0], -vf[:, :, 1], 0.4, color='r')
    # quiver(X,Y,-v(:,:,2),-v(:,:,1), 0.6);
    #axis xy
    #axis equal
    #axis off
    #hold off
