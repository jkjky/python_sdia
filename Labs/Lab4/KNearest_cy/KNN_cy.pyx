import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPEint = np.intc
DTYPEfloat = np.float64

# Define numeric types
ctypedef cnp.float64_t DTYPEfloat_t
ctypedef cnp.int_t DTYPEint_t

cnp.import_array()

cdef cnp.ndarray[DTYPEfloat_t, ndim=2] compute_distances(
    DTYPEfloat_t[:] V,
    DTYPEfloat_t[:, :] data
):
    cdef Py_ssize_t n = data.shape[0]
    cdef Py_ssize_t i

    cdef cnp.ndarray[DTYPEfloat_t, ndim=2] out = np.empty((n, 2), dtype=DTYPEfloat)

    cdef double dx, dy

    for i in range(n):
        dx = data[i, 1] - V[0]
        dy = data[i, 2] - V[1]
        out[i, 0] = data[i, 0]          # label_i
        out[i, 1] = sqrt(dx*dx + dy*dy) # distance_i

    return out

def KNearest(int K, x, data):

    cdef Py_ssize_t n = data.shape[0]
    cdef Py_ssize_t d = 2 #only features not the label
    cdef cnp.ndarray[DTYPEfloat_t, ndim=2] dist = compute_distances(x,data)
    cdef cnp.ndarray[DTYPEint_t, ndim=1] near = np.argsort(dist[:,1]).astype(DTYPEint)[:K]
    cdef cnp.ndarray[DTYPEint_t, ndim=1] labels = data[near, 0].astype(DTYPEint)
    cdef cnp.ndarray[DTYPEint_t, ndim=1] counter = np.bincount(labels).astype(DTYPEint)
    return np.argmax(counter)


def Nearest_neighbors(data, train, int K=3):

    cdef Py_ssize_t n = data.shape[0]
    cdef list classification = []
    cdef Py_ssize_t i

    for i in range(n):
        classification.append(KNearest(K, data[i], train))

    return np.array(classification, dtype=DTYPEint)
