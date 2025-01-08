# -*- coding: utf-8 -*-


from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import scipy.sparse as sparse
import os
# import time
# import h5py


########################################################################################################################
# LEER DATOS
def load_sparse_csr(filename):
    # here we need to add .npz extension manually
    loader = np.load(filename + '.npz')
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
########################################################################################################################


# ###### AJUSTAR el DIA ##############
DIA = 1  # 1..13
# ####################################
SUFIJO = '_WV'+str(DIA)

DIM_LECTOR = 5
DIM_DOCUMENTO = 662
DIM_ANUNCIANTE = 37

DIM_SOURCE = DIM_LECTOR + DIM_DOCUMENTO
DIM_TARGET = DIM_ANUNCIANTE + DIM_DOCUMENTO

K = 256  # The number of rows of matrices W and V (en el artículo)
DESV = 0.1
beta = 1e-5  # The regularization meta-parameter ν (en el artículo)
batch_size = 512

print('________________________________________________________\n')
print('TRABAJANDO CON DIA %d' % DIA)
print('________________________________________________________')

print('va a cargar...')

PATH = "./"

CDU = load_sparse_csr(PATH+"SPARSE_CASOS_USO_DAY"+str(DIA))
print(CDU.shape)
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
else:  # día 13
    CUANTOS = 4750000

CDU_train = CDU[0:CUANTOS, :]  # TRAIN
CDU_test = CDU[CUANTOS:, :]  # TEST
print(CDU_train.shape)
print(CDU_test.shape)

NE_train, _ = CDU_train.shape
NE_test, _ = CDU_test.shape

ITER_EN_EPOCH_train = NE_train // batch_size
ITER_EN_EPOCH_test = NE_test // batch_size

sess = tf.compat.v1.InteractiveSession()

tf.compat.v1.disable_eager_execution()

# placeholders
lecdoc = tf.compat.v1.placeholder(tf.float32, [None, DIM_SOURCE], name="lectorDocumento")
difftargets = tf.compat.v1.placeholder(tf.float32, [None, DIM_TARGET], name="difftargets")
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
# merged = tf.compat.v1.summary.merge_all()
# train_writer = tf.compat.v1.summary.FileWriter('./TB'+SUFIJO, sess.graph)
tf.compat.v1.global_variables_initializer().run()

# Create a saver object which will save all the variables
saver = tf.compat.v1.train.Saver()

ckpt = tf.train.get_checkpoint_state(os.path.dirname(PATH+'modelOB'+SUFIJO+'/checkpoint'))
# if that checkpoint exists, restore from checkpoint
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)

fallos = probs <= 0.5
num_fallos = tf.reduce_sum(tf.cast(fallos, tf.float32))


ITER = 0
EPOCHS = 130
total = EPOCHS * ITER_EN_EPOCH_train
for ep in range(EPOCHS):
    for valStep in range(0, ITER_EN_EPOCH_train):
        # index of examples in batch
        start_batch = valStep * batch_size
        if valStep == ITER_EN_EPOCH_train-1:
            end_batch = NE_train - 1
            clase_train = np.ones((end_batch-start_batch, 1))
        else:
            end_batch = valStep*batch_size+batch_size
            clase_train = np.ones((batch_size, 1))
        thisIdx = list(range(start_batch, end_batch))
        # examples in batch
        lecdoc_train = CDU_train[thisIdx, 0:DIM_SOURCE].todense()
        difftargets_train = CDU_train[thisIdx, DIM_SOURCE:].todense()
        # learning step
        # summary, vLoss, gs, _ = sess.run([merged, softplus_loss, global_step, train_step],
        vLoss, gs, _ = sess.run([softplus_loss, global_step, train_step],
                                         feed_dict={lecdoc: lecdoc_train,
                                                    difftargets: difftargets_train,
                                                    clase: clase_train,
                                                    keep_prob: 0.75})
        # print("epoch %4d, iter %6d/%d, global_step %d loss = %f" % (ep, ITER, total, gs, vLoss))
        if ITER % 3000 == 0:
            # print(thisIdx)
            print("epoch %4d, iter %6d/%d, global_step %d loss = %f" % (ep, ITER, total, gs, vLoss))

            # le paso la iteración global para poder continuar con la gráfica
            # train_writer.add_summary(summary, global_step=gs)

        if ITER % 100000 == 0:
            # guardar modelo al acabar la epoch
            saver.save(sess, PATH+'modelOB'+SUFIJO+'/model', global_step=global_step)

        ITER = ITER + 1

    if ep % 4 == 0:
        NF = 0
        for valStep in range(0, ITER_EN_EPOCH_train):
            thisIdx = list(range(valStep * batch_size, valStep * batch_size + batch_size))
            lecdoc_test = CDU_train[thisIdx, 0:DIM_SOURCE].todense()
            difftargets_test = CDU_train[thisIdx, DIM_SOURCE:].todense()
            nf_batch = sess.run(num_fallos, feed_dict={lecdoc: lecdoc_test,
                                                       difftargets: difftargets_test, keep_prob: 1})
            NF = NF + nf_batch
        print("TRAIN  epoch %d  global_step %d, NumFallos: %4d, error %.4f" % (ep, gs, NF, NF / NE_train))
        NF = 0
        for valStep in range(0, ITER_EN_EPOCH_test):
            thisIdx = list(range(valStep * batch_size, valStep * batch_size + batch_size))
            lecdoc_test = CDU_test[thisIdx, 0:DIM_SOURCE].todense()
            difftargets_test = CDU_test[thisIdx, DIM_SOURCE:].todense()
            nf_batch = sess.run(num_fallos, feed_dict={lecdoc: lecdoc_test,
                                                       difftargets: difftargets_test, keep_prob: 1})
            NF = NF + nf_batch
        print("TEST   epoch %d  global_step %d, NumFallos: %4d, error %.4f" % (ep, gs, NF, NF / NE_test))

# train_writer.close()

saver.save(sess, PATH+'modelOB'+SUFIJO+'/model')
