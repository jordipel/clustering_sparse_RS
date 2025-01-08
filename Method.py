import numpy as np
import scipy.sparse as sparse


def load_sparse_csr(filename):
    # here we need to add .npz extension manually
    loader = np.load(filename + '.npz')
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])



CDU = load_sparse_csr("SPARSE_CASOS_USO")