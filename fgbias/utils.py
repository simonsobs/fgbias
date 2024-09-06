import numpy as np
from pixell import curvedsky

def get_cl_smooth(alm1, alm2=None, n=5):
    if alm2 is None:
        alm2=alm1
    cl = curvedsky.alm2cl(alm1, alm2)
    window = np.ones(int(n))/float(n)
    cl_out = np.convolve(cl, window, mode="same")
    cl_out[-n:] = cl[-n:]
    return cl_out