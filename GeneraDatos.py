# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sqlite3 as lite
import sys, time, cPickle, feather, pickle
import os.path
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D
import collections
import random
import fileinput
import h5py

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

def load_sparse_csr(filename):
    # here we need to add .npz extension manually
    loader = np.load(filename + '.npz')
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

def getPickle(name):

    filehandler = open(name+".p", "rb")
    object = pickle.load( filehandler)
    filehandler.close()

    return object

def save_sparse_csr(filename, array):
    # note that .npz extension is added automatically
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-


for day in range(1,14):

    USR_RAW = load_sparse_csr("SPARSE_USUARIOS_DAY"+str(day))

    LABELS_FOURDOTTWO = getPickle("ClusteringResults/model_DAY"+str(day)+"_100_VALPOND_RAW_labels").transpose()
    LABELS_FOURDOTTWO =np.asmatrix(LABELS_FOURDOTTWO).transpose()

    USR_RAW = USR_RAW.todense()
    USR = np.zeros((USR_RAW.shape[0],USR_RAW.shape[1]+1))

    USR[:,0:USR_RAW.shape[1]] = USR_RAW
    USR[:,USR_RAW.shape[1]:] = LABELS_FOURDOTTWO

    save_sparse_csr("DatosGalicia/DATOS_DIA"+str(day), sparse.csr_matrix(USR))

    print("Dia "+str(day)+" generado.")

