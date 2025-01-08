# -*- coding: utf-8 -*-


# from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import scipy.sparse as sparse
import os
import time
import pickle
import h5py


########################################################################################################################
# LEER DATOS


def load_sparse_csr(filename):
    # here we need to add .npz extension manually
    loader = np.load(filename + '.npz')
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
########################################################################################################################


# ###### AJUSTAR el DIA ##############
DIA = 1  # 1..13
#####################################
SUFIJO = '_WV'+str(DIA)


DIM_LECTOR = 5
DIM_DOCUMENTO = 662
DIM_ANUNCIANTE = 37

DIM_SOURCE = DIM_LECTOR + DIM_DOCUMENTO
DIM_TARGET = DIM_ANUNCIANTE + DIM_DOCUMENTO

K = 256
DESV = 0.1
beta = 1e-5
batch_size = 2048  # 512

print('________________________________________________________\n')
print('TRABAJANDO CON DIA %d' % DIA)
print('________________________________________________________')

print('va a cargar...')

COMPUTE_VAL_CENTROIDS = True  # para calcular el valor de
# COMPUTE_VAL_CENTROIDS = False

TRAIN = True
#TRAIN = False

# COMPUTE_ERROR = True
COMPUTE_ERROR = False

# COMPUTE_PROBS = True
COMPUTE_PROBS = False

COMPUTE_EMBEDDING = True
# COMPUTE_EMBEDDING = False

sess = tf.compat.v1.InteractiveSession()

tf.compat.v1.disable_eager_execution()

# placeholders
lecdoc = tf.compat.v1.placeholder(tf.float32, [None, DIM_SOURCE], name="lectorDocumento")
difftargets = tf.compat.v1.placeholder(tf.float32, [None, DIM_TARGET], name="difftargets")
cent_emb = tf.compat.v1.placeholder(tf.float32, [None, K], name="cent")
clase = tf.compat.v1.placeholder(tf.float32, [None, 1], name="clase")
keep_prob = tf.compat.v1.placeholder(tf.float32, name="keep_prob")

# global step
global_step = tf.compat.v1.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

with tf.name_scope('f_lecdoc'):
    n_input = DIM_SOURCE
    n_output = K
    W = tf.compat.v1.Variable(tf.random.normal([n_input, n_output], stddev=DESV), name="W")
    f_lecdoc = tf.matmul(lecdoc, W)

with tf.name_scope('f_diffs'):
    n_input = DIM_TARGET
    n_output = K
    V = tf.compat.v1.Variable(tf.random.normal([n_input, n_output], stddev=DESV), name="V")
    f_diffs = tf.matmul(difftargets, V)

with tf.name_scope('prod_escalar_con_softplus'):
    output = tf.reduce_sum(tf.multiply(f_lecdoc, f_diffs), 1, keepdims=True, name="prod_esc")
    # output_cent_emb = tf.reduce_sum(tf.multiply(f_lecdoc, cent_emb), 1, keep_dims=True, name="prod_esc_cent")

# las probabilidades son la sigmoide de la salida
probs = tf.nn.sigmoid(output)

with tf.name_scope('funcion_perdida'):
    regularizers = tf.nn.l2_loss(W) + tf.nn.l2_loss(V)
    softplus_loss = tf.reduce_mean(tf.nn.softplus((1 - (2*clase)) * output), name="soft_plus")
    softplus_loss = tf.reduce_mean(softplus_loss + beta * regularizers)

tf.summary.scalar('softplus_loss', softplus_loss)

# optimizer
with tf.name_scope('train'):
    train_step = tf.compat.v1.train.AdamOptimizer(0.0001).minimize(softplus_loss, global_step=global_step)

# Merge all the summaries and write them out
# merged = tf.summary.merge_all()
# train_writer = tf.summary.FileWriter('./TB'+SUFIJO, sess.graph)
tf.compat.v1.global_variables_initializer().run()

# Create a saver object which will save all the variables
saver = tf.compat.v1.train.Saver()

ckpt = tf.train.get_checkpoint_state(os.path.dirname('MODELS/modelOB'+SUFIJO+'/checkpoint'))
# if that checkpoint exists, restore from checkpoint
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("CARGADO:", ckpt.model_checkpoint_path)
else:
    print('No encuentro el modelo')
    exit()

fallos = probs <= 0.5
num_fallos = tf.reduce_sum(tf.cast(fallos, tf.float32))

if COMPUTE_VAL_CENTROIDS:
    CDU = load_sparse_csr("SPARSES/SPARSE_USUARIOS_DAY" + str(DIA))
    NE, _ = CDU.shape
    ITER_EN_EPOCH = NE // batch_size
    print(CDU.shape)

    if TRAIN:
        print('TRAIN')
        centroides = pickle.load(open('DIFFS/DIFF_RAW_CENTROIDS_TRAIN_DAY' + str(DIA) + '.p', 'rb'),
                                 encoding='latin1')
    else:
        print('TEST')
        centroides = pickle.load(open('DIFFS/DIFF_RAW_CENTROIDS_TEST_DAY' + str(DIA) + '.p', 'rb'),
                                 encoding='latin1')
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
                end_batch = NE - 1
            else:
                end_batch = valStep * batch_size + batch_size
            thisIdx = list(range(start_batch, end_batch))
            # examples in batch
            lecdoc_test = CDU[thisIdx, 0:DIM_SOURCE].todense()
            cent = np.tile(centroides[c, :], [len(thisIdx), 1])  # esto es como el repmat de matlab
            ld_batch = sess.run(output, feed_dict={lecdoc: lecdoc_test, difftargets: cent, keep_prob: 1})
            #    ld_batch = sess.run(output_cent_emb, feed_dict={lecdoc: lecdoc_test, cent_emb: cent, keep_prob: 1})
            VAL_POR_CENTROIDE[start_batch:end_batch, c:(c+1)] = ld_batch
            if valStep % 100 == 0:
                print('%d -> %d' % (c, end_batch))
    t = time.time()
    if TRAIN:
        fileh5 = h5py.File('VALORACIONES/VAL_POR_CENTROIDE_RAW_TRAIN_DAY' + str(DIA) + '.h5', 'w')
    else:
        fileh5 = h5py.File('VALORACIONES/VAL_POR_CENTROIDE_RAW_TEST_DAY' + str(DIA) + '.h5', 'w')
    fileh5.create_dataset('VAL_POR_CENTROIDE', data=VAL_POR_CENTROIDE)
    fileh5.close()
    print("%.4f" % (time.time() - t))
    print(VAL_POR_CENTROIDE[0:10, :])

if COMPUTE_PROBS:
    CDU = load_sparse_csr("SPARSE_CASOS_USO_DAY" + str(DIA))
    NE, _ = CDU.shape
    ITER_EN_EPOCH = NE // batch_size
    PROBS = np.empty([NE, 1])
    t = time.time()
    for valStep in range(0, ITER_EN_EPOCH):
        # index of examples in batch
        start_batch = valStep * batch_size
        if valStep == ITER_EN_EPOCH - 1:
            end_batch = NE - 1
        else:
            end_batch = valStep * batch_size + batch_size
        thisIdx = list(range(start_batch, end_batch))
        # examples in batch
        lecdoc_test = CDU[thisIdx, 0:DIM_SOURCE].todense()
        difftargets_test = CDU[thisIdx, DIM_SOURCE:].todense()

        ld_batch = sess.run(probs, feed_dict={lecdoc: lecdoc_test, difftargets: difftargets_test, keep_prob: 1})
        PROBS[start_batch:end_batch, :] = ld_batch
        if valStep % 100 == 0:
            print(end_batch)

    t = time.time()
    fileh5 = h5py.File('PROBABILIDADES_DAY'+str(DIA)+'.h5', 'w')
    fileh5.create_dataset('probabilidades', data=PROBS)
    fileh5.close()
    print("%.4f" % (time.time() - t))


# ERROR ################################################################################################################

if COMPUTE_ERROR:
    CDU = load_sparse_csr("SPARSE_CASOS_USO_DAY" + str(DIA))
    NE, _ = CDU.shape
    ITER_EN_EPOCH = NE // batch_size
    t = time.time()
    NF = 0
    for valStep in range(0, ITER_EN_EPOCH):
        # index of examples in batch
        start_batch = valStep * batch_size
        if valStep == ITER_EN_EPOCH - 1:
            end_batch = NE - 1
        else:
            end_batch = valStep * batch_size + batch_size
        thisIdx = list(range(start_batch, end_batch))
        # examples in batch
        lecdoc_test = CDU[thisIdx, 0:DIM_SOURCE].todense()
        difftargets_test = CDU[thisIdx, DIM_SOURCE:].todense()

        nf_batch = sess.run(num_fallos, feed_dict={lecdoc: lecdoc_test,
                                                   difftargets: difftargets_test, keep_prob: 1})

        NF = NF + nf_batch
        if valStep % 100 == 0:
            print(end_batch)

    print("NumFallos: %4d, error %.4f" % (NF, NF / NE))
    print("%.4f" % (time.time() - t))



# COMPUTE_EMBEDDING ####################################################################################################

if COMPUTE_EMBEDDING:
    CDU = load_sparse_csr("SPARSES/SPARSE_USUARIOS_DAY" + str(DIA))
    NE, _ = CDU.shape
    ITER_EN_EPOCH = NE // batch_size

    EMB = np.empty([NE, K])
    t = time.time()
    for valStep in range(0, ITER_EN_EPOCH):
        # index of examples in batch
        start_batch = valStep * batch_size
        if valStep == ITER_EN_EPOCH - 1:
            end_batch = NE - 1
        else:
            end_batch = valStep * batch_size + batch_size
        thisIdx = list(range(start_batch, end_batch))
        # examples in batch
        lecdoc_test = CDU[thisIdx, 0:DIM_SOURCE].todense()

        ld_batch = sess.run(f_lecdoc, feed_dict={lecdoc: lecdoc_test, keep_prob: 1})
        EMB[start_batch:end_batch, :] = ld_batch
        if valStep % 100 == 0:
            print(end_batch)

    t = time.time()
    fileh5 = h5py.File('EMBEDDINGS/EMBEDDING_USUARIOS_DAY'+str(DIA)+'.h5', 'w')
    fileh5.create_dataset('embedding', data=EMB)
    fileh5.close()
    print("%.4f" % (time.time()-t))

    ##############################################################
    '''CDU = load_sparse_csr("SPARSE_ANUNCIOS_DAY" + str(DIA))
    NE, _ = CDU.shape
    print(CDU.shape)
    ITER_EN_EPOCH = NE // batch_size

    EMB = np.empty([NE, K])
    t = time.time()
    for valStep in range(0, ITER_EN_EPOCH):
        # index of examples in batch
        start_batch = valStep * batch_size
        if valStep == ITER_EN_EPOCH - 1:
            end_batch = NE - 1
        else:
            end_batch = valStep * batch_size + batch_size
        thisIdx = list(range(start_batch, end_batch))
        # examples in batch
        difftargets_test = CDU[thisIdx, 0:DIM_TARGET].todense()

        dt_batch = sess.run(f_diffs, feed_dict={difftargets: difftargets_test, keep_prob: 1})
        EMB[start_batch:end_batch, :] = dt_batch
        if valStep % 100 == 0:
            print(end_batch)

    t = time.time()
    fileh5 = h5py.File('EMBEDDING_ANUNCIOS_DAY'+str(DIA)+'.h5', 'w')
    fileh5.create_dataset('embedding', data=EMB)
    fileh5.close()
    print("%.4f" % (time.time() - t))

    ##############################################################
    CDU = load_sparse_csr("SPARSE_CASOS_USO_DAY" + str(DIA))
    NE, _ = CDU.shape
    ITER_EN_EPOCH = NE // batch_size
    EMB = np.empty([NE, K])
    t = time.time()
    for valStep in range(0, ITER_EN_EPOCH):
        # index of examples in batch
        start_batch = valStep * batch_size
        if valStep == ITER_EN_EPOCH - 1:
            end_batch = NE - 1
        else:
            end_batch = valStep * batch_size + batch_size
        thisIdx = list(range(start_batch, end_batch))
        # examples in batch
        difftargets_test = CDU[thisIdx, DIM_SOURCE:].todense()

        dt_batch = sess.run(f_diffs, feed_dict={difftargets: difftargets_test, keep_prob: 1})
        EMB[start_batch:end_batch, :] = dt_batch
        if valStep % 100 == 0:
            print(end_batch)

    t = time.time()
    fileh5 = h5py.File('EMBEDDING_DIFFS_DAY' + str(DIA) + '.h5', 'w')
    fileh5.create_dataset('embedding', data=EMB)
    fileh5.close()
    print("%.4f" % (time.time() - t))'''


print("FIN!")
