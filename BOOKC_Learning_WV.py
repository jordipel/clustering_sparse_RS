# -*- coding: utf-8 -*-


import tensorflow as tf
import os
import pickle
from keras.utils import to_categorical

# LEEME
#
# UTILIZO LA REPRESENTACION ONE HOT DE LOS USUARIOS (NO UTILIZO LOS DATOS CONOCIDOS DE LOS USUARIOS)
# Y LA REPRESENTACIÓN DE LAS PELÍCULAS POR SUS GÉNEROS
# LAS PELÍCULAS NO SE PASAN A ONEHOT PORQUE HE SEPARADO PELÍCULAS PARA TEST
# QUE NO ESTÁN EN EL TRAIN
#
#

########################################################################################################################
# LEER DATOS
def getPickle(name):
    filehandler = open(name, "rb")
    object = pickle.load(filehandler)
    filehandler.close()

    return object
########################################################################################################################

PREFIJO = 'BOOKC'

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
print('TRABAJANDO CON ' + PREFIJO)
print('________________________________________________________')

print('va a cargar...')

CDU_train = getPickle('SPARSES/'+PREFIJO+'_CASOS_USO_TRAIN_con_ids.pkl')
CDU_test = getPickle('SPARSES/'+PREFIJO+'_CASOS_USO_TEST_con_ids.pkl')
print(CDU_train.shape)
print(CDU_test.shape)

# lo convierto en un problema de clasificación para luego poder calcular incertidumbres
CDU_train.loc[CDU_train.rating < 7.5, 'rating'] = 0   # no gusta
CDU_train.loc[CDU_train.rating >= 7.5, 'rating'] = 1  # gusta

CDU_test.loc[CDU_test.rating < 7.5, 'rating'] = 0   # no gusta
CDU_test.loc[CDU_test.rating >= 7.5, 'rating'] = 1  # gusta

print("No gusta en test:", CDU_test.loc[CDU_test.rating == 0].shape[0] / CDU_test.shape[0] * 100)
print("Gusta en test:", CDU_test.loc[CDU_test.rating == 1].shape[0] / CDU_test.shape[0] * 100)

# calculo el one-hot de los usuarios
# oh_user_train = to_categorical(CDU_train.user_id, dtype='int32')
# oh_user_test = to_categorical(CDU_test.user_id, dtype='int32')

NE_train, _ = CDU_train.shape
NE_test, _ = CDU_test.shape

ITER_EN_EPOCH_train = NE_train // batch_size
ITER_EN_EPOCH_test = NE_test // batch_size

sess = tf.compat.v1.InteractiveSession()

tf.compat.v1.disable_eager_execution()

# placeholders
# lecdoc = tf.compat.v1.placeholder(tf.float32, [None, DIM_USER_ONE_HOT], name="lectorDocumento")
lecdoc = tf.compat.v1.placeholder(tf.float32, [None, DIM_USER], name="lectorDocumento")
difftargets = tf.compat.v1.placeholder(tf.float32, [None, DIM_MOVIE], name="difftargets")
clase = tf.compat.v1.placeholder(tf.float32, [None, 1], name="clase")
keep_prob = tf.compat.v1.placeholder(tf.float32, name="keep_prob")

# global step
global_step = tf.compat.v1.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

with tf.name_scope('f_lecdoc'):
    # n_input = DIM_USER_ONE_HOT
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

# fallos = probs <= 0.5
fallos = (output * (clase*2 - 1)) < 0
num_fallos = tf.reduce_sum(tf.cast(fallos, tf.float32))


ITER = 0
EPOCHS = 1001  # 130
total = EPOCHS * ITER_EN_EPOCH_train
for ep in range(EPOCHS):
    for valStep in range(0, ITER_EN_EPOCH_train):
        # index of examples in batch
        start_batch = valStep * batch_size
        if valStep == ITER_EN_EPOCH_train-1:
            end_batch = NE_train - 1
        else:
            end_batch = valStep*batch_size+batch_size
        thisIdx = list(range(start_batch, end_batch))
        # examples in batch
        lecdoc_train = CDU_train.iloc[thisIdx, 3:(DIM_USER+3)]
        # lecdoc_train = oh_user_train[thisIdx, :]
        difftargets_train = CDU_train.iloc[thisIdx, (DIM_USER+3):]
        clase_train = CDU_train.iloc[thisIdx, 2].values.reshape(len(thisIdx), 1)
        # learning step
        # summary, vLoss, gs, _ = sess.run([merged, softplus_loss, global_step, train_step],
        # vLoss, gs, _ = sess.run([MSE_loss, global_step, train_step],
        vLoss, gs, _ = sess.run([softplus_loss, global_step, train_step],
                                feed_dict={lecdoc: lecdoc_train,
                                                    difftargets: difftargets_train,
                                                    clase: clase_train,
                                                    keep_prob: KP_train})
        # print("epoch %4d, iter %6d/%d, global_step %d loss = %f" % (ep, ITER, total, gs, vLoss))
        # (ITER_EN_EPOCH_train // 50) <- esto sería cada 2% de la epoch (aproximadamente)
        # (ITER_EN_EPOCH_train // 100) <- esto sería cada 1% de la epoch (aproximadamente)
        if ITER % (ITER_EN_EPOCH_train // 5) == 0:
            # print(thisIdx)
            print("%.2f%%" % ((ITER % ITER_EN_EPOCH_train) * 100 / ITER_EN_EPOCH_train), end=' ')
            print("epoch %4d, iter %6d/%d, global_step %d loss = %f" % (ep, ITER, total, gs, vLoss))

        # (ITER_EN_EPOCH_train // 5) <- esto sería cada 20% de la epoch (aproximadamente)
        if ITER % ITER_EN_EPOCH_train == 0:  # cada epoch
            saver.save(sess, 'MODELS/' + PREFIJO + '_model_WV/model', global_step=global_step)

        ITER = ITER + 1

    if ep % 2 == 0:
        NF = 0
        for valStep in range(0, ITER_EN_EPOCH_train):
            thisIdx = list(range(valStep * batch_size, valStep * batch_size + batch_size))
            lecdoc_train = CDU_train.iloc[thisIdx, 3:(DIM_USER + 3)]
            # lecdoc_train = oh_user_train[thisIdx, :]
            difftargets_train = CDU_train.iloc[thisIdx, (DIM_USER + 3):]
            clase_train = CDU_train.iloc[thisIdx, 2].values.reshape(len(thisIdx), 1)
            nf_batch = sess.run(num_fallos, feed_dict={lecdoc: lecdoc_train,
                                                       difftargets: difftargets_train,
                                                       clase: clase_train, keep_prob: 1})
            NF = NF + nf_batch
            # if valStep % (ITER_EN_EPOCH_train // 100) == 0:
            #    print("%.2f%%" % ((valStep % ITER_EN_EPOCH_train) * 100 / ITER_EN_EPOCH_train), end=' ')
        print("TRAIN  epoch %d  global_step %d, NumFallos: %4d, error %.4f" % (ep, gs, NF, NF / NE_train))
        NF = 0
        for valStep in range(0, ITER_EN_EPOCH_test):
            thisIdx = list(range(valStep * batch_size, valStep * batch_size + batch_size))
            lecdoc_test = CDU_test.iloc[thisIdx, 3:(DIM_USER + 3)]
            # lecdoc_test = oh_user_test[thisIdx, :]
            difftargets_test = CDU_test.iloc[thisIdx, (DIM_USER + 3):]
            clase_test = CDU_test.iloc[thisIdx, 2].values.reshape(len(thisIdx), 1)
            nf_batch = sess.run(num_fallos, feed_dict={lecdoc: lecdoc_test,
                                                       difftargets: difftargets_test,
                                                       clase: clase_test, keep_prob: 1})
            NF = NF + nf_batch
            # if valStep % (ITER_EN_EPOCH_train // 100) == 0:
            #    print("%.2f%%" % ((valStep % ITER_EN_EPOCH_train) * 100 / ITER_EN_EPOCH_train), end=' ')
        print("TEST   epoch %d  global_step %d, NumFallos: %4d, error %.4f" % (ep, gs, NF, NF / NE_test))


saver.save(sess, 'MODELS/'+PREFIJO+'_model_WV/model', global_step=global_step)

