# test for tensor voting
# Use the tensor structure direction field as an input.


import numpy as np
import matplotlib.pyplot as plt

import compute_gaussian_filter as gf
import perform_convolution as con
import compute_rigidity_tensor as crt
import compute_stick_tf as cst
import perform_tensor_decomp as ptd
import perform_tensor_recomp as ptr
import perform_voting as pv
import perform_vf_reorientation as pvr
import compute_grad as cg
import prod_vf_vf as pvv
import perform_curv_extraction as pce
import plot_vf as plv
import plot_curve as plc


sigma = 0.5
mu = 3

############################################
# a test image
print('--> Building test image.')

radius = 0.9

n = 75
x = np.linspace(-1, 1, n)
y = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, x)

N = abs(X ** 2 + y ** 2 - radius ** 2) < 0.03
N = (X ** 2 + Y ** 2) < radius ** 2

M = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if N[i, j]:
            M[i, j] = 1


# add some noise in the middle of the image
sigma = 0.8    # level of the noise
r = 0.05       # width of the gap
# I = find(abs(X - 0.5) < r)
#M = M + sigma * np.random.rand(n, n) * (abs(X - 0.5) < r)
for i in range(n):
    for j in range(n):
        if abs(X[i, j] - 0.5) < r:
            M[i, j] += sigma * np.random.normal()


#############################################
###### structure tensor computation
print('--> Compute tensor structure.')

h = gf.compute_gaussian_filter([11, 11], [0.6/n], [n, n])
Mh = con.perform_convolution(M, h)

ST = crt.compute_rigidity_tensor(Mh)

h = gf.compute_gaussian_filter([11, 11], [0.3/n], [n, n])
for i in range(2):   # convole each entry
    for j in range(2):
        ST[:, :, i, j] = con.perform_convolution( ST[:, :, i, j] ,h)

[e1, e2, l1, l2] = ptd.perform_tensor_decomp(ST)   # e1 is a regularized gradient

# As the main direction is the direction orthogonal to the discontinuity 
# we must swap the eigenvectors.
T = ptr.perform_tensor_recomp(e2, e1, l1, l2)

fig, axs = plt.subplots(2, 2)
plv.plot_vf(axs[0, 0], e2, M, True)
axs[0, 0].set_title('Original Image and Field')

#####################################################################
### an example of field
A = cst.compute_stick_tf([1, 0.3], n)
[e1, e2, l1, l2] = ptd.perform_tensor_decomp(A)

plv.plot_vf(axs[0, 1], e1, l1);
axs[0, 1].set_title('Example of Stick Field');

#####################################################################
#### peform voting
print('--> Performing tensor voting.')
sigma = 0.2
c = 0.1 * sigma
#exit()
T = pv.perform_voting(T, sigma, c)
print('--> Performing tensor decomposition.')
[e1, e2, l1, l2] = ptd.perform_tensor_decomp(T, 'abs')

# re-orient vf
#e1 = pvr.perform_vf_reorientation(e1)
#e2 = pvr.perform_vf_reorientation(e2)
plv.plot_vf(axs[1, 0], e1, l1)
axs[1, 0].set_title('New Field with Eigenvalue')

#####################################################################
#### curve extraction via zeros crossing of A=ds/dn where s=l1-l2, n=e2,
# orthogonal to e1
s = l1 - l2
g = cg.compute_grad(s)
A = pvv.prod_vf_vf(g, e2)

# extract only the 20 longest curves
#options.min_length = 20
c_list = pce.perform_curve_extraction(A, 0, options='min_length', leng=20)

#plt.show()
axs[1, 1].clear()

# plot curves
#subplot(2,2,4);
plc.plot_curve(axs[1, 1], c_list, A)
#axis off;
axs[1, 1].set_title('Extracted Curves')
axs[1, 1].set_aspect(1)
#saveas(gcf, 'test_structure', 'png')


plt.show()


