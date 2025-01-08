# -*- coding: utf-8 -*-


# from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
# import scipy.sparse as sparse
import os
import time
import pickle
import h5py
from keras.utils import to_categorical


PREFIJO = 'BOOKC'

########################################################################################################################
# LEER DATOS


def getPickle(name):
    filehandler = open(name, "rb")
    obj = pickle.load(filehandler)
    filehandler.close()
    return obj
########################################################################################################################


DIM_USER = 2618  #
# DIM_USER_ONE_HOT = 6041  # porque las películas están codificadas de 1 a 6040 (una dim más por el 0)
DIM_MOVIE = 2363  #


K = 256  # The number of rows of matrices W and V (en el artículo)
DESV = 0.1
beta = 0  # 1e-5  # The regularization meta-parameter ν (en el artículo)
batch_size = 512
KP_train = 1  # 0.75  # keep_prob en train (en test siempre = 1)
LR = 1e-7  # 1000 epochs con 1e-6

print('________________________________________________________\n')
print('TRABAJANDO CON', PREFIJO)
print('________________________________________________________')

print('va a cargar...')

# CDU_train = getPickle('SPARSES/'+PREFIJO+'_CASOS_USO_TRAIN_con_ids.pkl')
# CDU_test = getPickle('SPARSES/'+PREFIJO+'_CASOS_USO_TEST_con_ids.pkl')
# print(CDU_train.shape)
# print(CDU_test.shape)

COMPUTE_VAL_CENTROIDS = True  # para calcular el valor de
# COMPUTE_VAL_CENTROIDS = False

# TRAIN = True
TRAIN = False

COMPUTE_EMBEDDING = True
# COMPUTE_EMBEDDING = False

sess = tf.compat.v1.InteractiveSession()

tf.compat.v1.disable_eager_execution()

# placeholders
lecdoc = tf.compat.v1.placeholder(tf.float32, [None, DIM_USER], name="lectorDocumento")
difftargets = tf.compat.v1.placeholder(tf.float32, [None, DIM_MOVIE], name="difftargets")
cent_emb = tf.compat.v1.placeholder(tf.float32, [None, K], name="cent")
clase = tf.compat.v1.placeholder(tf.float32, [None, 1], name="clase")
keep_prob = tf.compat.v1.placeholder(tf.float32, name="keep_prob")


# global step
global_step = tf.compat.v1.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

with tf.name_scope('f_lecdoc'):
    n_input = DIM_USER
    n_output = K
    W = tf.compat.v1.Variable(tf.random.normal([n_input, n_output], stddev=DESV), name="W")
    f_lecdoc = tf.matmul(lecdoc, W)

with tf.name_scope('f_diffs'):
    n_input = DIM_MOVIE
    n_output = K
    V = tf.compat.v1.Variable(tf.random.normal([n_input, n_output], stddev=DESV), name="V")
    f_diffs = tf.matmul(difftargets, V)

with tf.name_scope('prod_escalar'):
    output = tf.reduce_sum(tf.multiply(f_lecdoc, f_diffs), 1, keepdims=True, name="prod_esc")
    # output_cent_emb = tf.reduce_sum(tf.multiply(f_lecdoc, cent_emb), 1, keep_dims=True, name="prod_esc_cent")

# las probabilidades son la sigmoide de la salida
probs = tf.nn.sigmoid(output)

with tf.name_scope('funcion_perdida'):
    regularizers = tf.nn.l2_loss(W) + tf.nn.l2_loss(V)
    softplus_loss = tf.reduce_mean(tf.nn.softplus((1 - (2*clase)) * output), name="soft_plus")
    softplus_loss = tf.reduce_mean(softplus_loss + beta * regularizers)

# optimizer
with tf.name_scope('train'):
    train_step = tf.compat.v1.train.AdamOptimizer(LR).minimize(softplus_loss, global_step=global_step)

tf.compat.v1.global_variables_initializer().run()

# Create a saver object which will save all the variables
saver = tf.compat.v1.train.Saver()

ckpt = tf.train.get_checkpoint_state(os.path.dirname('MODELS/'+PREFIJO+'_model_WV/checkpoint'))
# if that checkpoint exists, restore from checkpoint
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("CARGADO:", ckpt.model_checkpoint_path)
else:
    print('No encuentro el modelo')
    exit()

# fallos = probs <= 0.5
fallos = (output * (clase*2 - 1)) < 0
num_fallos = tf.reduce_sum(tf.cast(fallos, tf.float32))

if COMPUTE_VAL_CENTROIDS:

    CDU = getPickle('SPARSES/'+PREFIJO+'_USUARIOS.pkl')
    NE, _ = CDU.shape
    ITER_EN_EPOCH = NE // batch_size
    # CDU = to_categorical(range(NE + 1))
    # CDU = CDU[1:, :]  # quito el 0 porque no hay usuario 0
    print(CDU.shape)

    if TRAIN:
        print('TRAIN')
        centroides = getPickle('DIFFS/'+PREFIJO+'_BOOKS_RAW_CENTROIDS_TRAIN.pkl')
    else:
        print('TEST')
        centroides = getPickle('DIFFS/'+PREFIJO+'_BOOKS_RAW_CENTROIDS_TEST.pkl')

    print(centroides.shape)
    # centroides = np.array(centroides)
    NC, _ = centroides.shape
    VAL_POR_CENTROIDE = np.empty([NE, NC])
    t = time.time()
    for c in range(NC):
        for valStep in range(0, ITER_EN_EPOCH):
            # index of examples in batch
            start_batch = valStep * batch_size
            if valStep == ITER_EN_EPOCH - 1:
                end_batch = NE
            else:
                end_batch = valStep * batch_size + batch_size
            thisIdx = list(range(start_batch, end_batch))
            # print(thisIdx, len(thisIdx))
            # examples in batch
            lecdoc_test = CDU.iloc[thisIdx, :]
            cent = np.tile(centroides[c, :], [len(thisIdx), 1])  # esto es como el repmat de matlab
            ld_batch = sess.run(output, feed_dict={lecdoc: lecdoc_test, difftargets: cent, keep_prob: 1})
            #    ld_batch = sess.run(output_cent_emb, feed_dict={lecdoc: lecdoc_test, cent_emb: cent, keep_prob: 1})
            VAL_POR_CENTROIDE[start_batch:end_batch, c:(c+1)] = ld_batch
            if valStep % 100 == 0:
                print('%d -> %d' % (c, end_batch))
    t = time.time()
    if TRAIN:
        fileh5 = h5py.File('VALORACIONES/'+PREFIJO+'_VAL_POR_CENTROIDE_RAW_TRAIN.h5', 'w')
    else:
        fileh5 = h5py.File('VALORACIONES/'+PREFIJO+'_VAL_POR_CENTROIDE_RAW_TEST.h5', 'w')
    fileh5.create_dataset('VAL_POR_CENTROIDE', data=VAL_POR_CENTROIDE)
    fileh5.close()
    print("%.4f" % (time.time() - t))
    print(VAL_POR_CENTROIDE[0:10, :])


# COMPUTE_EMBEDDING ####################################################################################################

if COMPUTE_EMBEDDING:
    CDU = getPickle('SPARSES/'+PREFIJO+'_USUARIOS.pkl')
    NE, _ = CDU.shape
    ITER_EN_EPOCH = NE // batch_size
    # CDU = to_categorical(range(NE + 1))
    # CDU = CDU[1:, :]  # quito el 0 porque no hay usuario 0
    print(CDU.shape)

    EMB = np.empty([NE, K])
    t = time.time()
    for valStep in range(0, ITER_EN_EPOCH):
        # index of examples in batch
        start_batch = valStep * batch_size
        if valStep == ITER_EN_EPOCH - 1:
            end_batch = NE
        else:
            end_batch = valStep * batch_size + batch_size
        thisIdx = list(range(start_batch, end_batch))
        # examples in batch
        lecdoc_test = CDU.iloc[thisIdx, :]

        ld_batch = sess.run(f_lecdoc, feed_dict={lecdoc: lecdoc_test, keep_prob: 1})
        EMB[start_batch:end_batch, :] = ld_batch
        if valStep % 100 == 0:
            print(end_batch)

    t = time.time()
    fileh5 = h5py.File('EMBEDDINGS/'+PREFIJO+'_EMBEDDING_USUARIOS.h5', 'w')
    fileh5.create_dataset('embedding', data=EMB)
    fileh5.close()
    print("%.4f" % (time.time()-t))

    ##############################################################


print("FIN!")
