# -*- coding: utf-8 -*-


from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import random
import os


# generar el conjunto de ejemplos
# cuatro atributos cada ejemplo
# atributo 1: valores 1, 2 o 3 con la misma probabilidad
# atributo 2: valores 1, 2 o 3 con la misma probabilidad
# atributo 3: valores 1 (98%), 2 (1%) o 3 (1%)
# atributo 4: valores 1 (90%), 2 (5%) o 3 (5%)
NE = 10000
DATOS = np.zeros((NE, 4))
# for i in range(NE):
#     DATOS[i,0] = random.randint(1,3)
#     DATOS[i,1] = random.randint(1,3)
#     val3 = random.random()
#     if val3 < 0.98:
#         DATOS[i,2] = 1
#     elif val3 < 0.99:
#         DATOS[i, 2] = 2
#     else:
#         DATOS[i, 2] = 3
#     val4 = random.random()
#     if val4 < 0.90:
#         DATOS[i, 3] = 1
#     elif val4 < 0.95:
#         DATOS[i, 3] = 2
#     else:
#         DATOS[i, 3] = 3
for i in range(NE):
    val = random.random()
    if val < 0.5:
        DATOS[i, 0] = 1
    val = random.random()
    if val < 0.30:
        DATOS[i, 1] = 1
    val = random.random()
    if val < 0.05:
        DATOS[i, 2] = 1
    val = random.random()
    if val < 0.02:
        DATOS[i, 3] = 1


# Parameters
display_step = 1
examples_to_show = 15
batch_size = 256

valIdx = list(range(0, NE))
repeatVals = int(np.floor(NE/batch_size))

# Network Parameters
n_input = 4  #
n_hidden_1 = 20  # 1st layer num features
n_hidden_2 = 2  # 2nd layer num features


# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}

keep_prob = tf.placeholder(tf.float32, name="keep_prob")

global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')


# Building the encoder
def encoder(x, keep_prob):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_1 = tf.nn.dropout(layer_1, keep_prob)  # dropout
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x, keep_prob):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_1 = tf.nn.dropout(layer_1, keep_prob)  # dropout
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X, keep_prob)
decoder_op = decoder(encoder_op, keep_prob)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
squared_error = tf.reduce_mean(tf.pow(y_true - y_pred, 2))

learning_rate = 0.05
# train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(squared_error)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(squared_error, global_step=global_step)
# train_step = tf.train.AdamOptimizer(learning_rate).minimize(squared_error)

# Create a saver object which will save all the variables
saver = tf.train.Saver()

# Initializing the variables
init = tf.global_variables_initializer()


ITER = 0
EPOCHS = 101
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('modelAE/checkpoint'))
    # if that checkpoint exists, restore from checkpoint
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    for ep in range(EPOCHS):
        for valStep in range(0, repeatVals):
            thisIdx = valIdx[valStep:NE:repeatVals]
            # print(thisIdx)
            x_batch = DATOS[thisIdx, :]
            _, EE = sess.run([train_step, squared_error], feed_dict={X: x_batch, keep_prob: 1})
            ITER = ITER + 1
        if ep % 10 == 0:
            print("ITER: %d\tsquared_error: %f" % (ITER, EE))
            # Now, save the graph
            saver.save(sess, 'modelAE/model', global_step=global_step)

    print("Optimization Finished!")

    # Now, save the graph
    saver.save(sess, 'modelAE/model')

    # Applying encode and decode over test set
    enc, dec = sess.run([encoder_op, y_pred], feed_dict={X: DATOS[:examples_to_show], keep_prob: 1})
    # Compare original images with their reconstructions
    for i in range(examples_to_show):
        print(DATOS[i], '->', enc[i], '->', dec[i])

    EE = sess.run(squared_error, feed_dict={X: DATOS, keep_prob: 1})
    print("FINAL\tsquared_error: %f" % EE)