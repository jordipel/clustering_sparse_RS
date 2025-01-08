# -*- coding: utf-8 -*-


from __future__ import division, print_function, absolute_import

import numpy as np
import scipy.sparse as sparse


########################################################################################################################
# LEER DATOS
def load_sparse_csr(filename):
    # here we need to add .npz extension manually
    loader = np.load(filename + '.npz')
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
########################################################################################################################

DIA_INI = 1
DIA_FIN = 13

for i in range(DIA_INI,DIA_FIN+1):
    DIA = i + DIA_INI - 1
    print('________________________________________________________\n')
    print('TRABAJANDO CON DIA %d' % DIA)

    CDU = load_sparse_csr("SPARSE_CASOS_USO_DAY" + str(DIA))
    print(CDU.shape)
    NE_total, _ = CDU.shape

    # exit()
    if DIA == 1:
        CUANTOS = 5500000
    elif DIA == 2:
        CUANTOS = 5000000
    elif DIA == 3:
        CUANTOS = 5250000
    elif DIA == 4:
        CUANTOS = 5000000
    elif DIA == 5:
        CUANTOS = 4250000
    elif DIA == 6:
        CUANTOS = 4250000
    elif DIA == 7:
        CUANTOS = 5500000
    elif DIA == 8:
        CUANTOS = 5250000
    elif DIA == 9:
        CUANTOS = 5250000
    elif DIA == 10:
        CUANTOS = 5250000
    elif DIA == 11:
        CUANTOS = 5250000
    elif DIA == 12:
        CUANTOS = 4500000
    elif DIA == 13:
        CUANTOS = 4750000

    CDU_train = CDU[0:CUANTOS, :]  # TRAIN
    CDU_test = CDU[CUANTOS:, :]  # TEST
    print(CDU_train.shape)
    print(CDU_test.shape)

    
    NE_train, _ = CDU_train.shape
    NE_test, _ = CDU_test.shape

    pctTrain = NE_train * 100 / NE_total
    pctTest = NE_test * 100 / NE_total

    print('95 por ciento = %d' % int(np.floor(NE_total * 0.95)))
    print('5 por ciento  = %d' % int(np.floor(NE_total*0.05)))

    print('__________DIA %d: TRAIN %.2f\tTEST %.2f__________' %(DIA, pctTrain, pctTest))

