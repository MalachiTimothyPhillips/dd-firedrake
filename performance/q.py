import numpy as np
def five_point_stencil(N,i,j):
    ip1j=4*N*j+4*i+4
    im1j=4*N*j+4*i+2
    ijp1=4*N*j+4*i+3
    ijm1=4*N*j+4*i+1
    return [ip1j,im1j,ijp1,ijm1]

# check a few cases quickly:

N=4
print(five_point_stencil(N,1,2))
# what it should be for five_point stencil, this particular case
print([40,38,39,37])
print(five_point_stencil(N,2,1))
print([28,26,27,25])
print(five_point_stencil(N,3,3))
print([64,62,63,61])
