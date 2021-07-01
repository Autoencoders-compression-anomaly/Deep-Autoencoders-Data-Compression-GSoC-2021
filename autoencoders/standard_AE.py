import pandas as pd
import tensorflow as tf
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


# Parameters
input_dim = 28
hidden_size1 = 100
hidden_size2 = 50
hidden_size3 = 30
z_dim = 20

batch_size = 100
n_epochs = 2
learning_rate = 0.001
beta1 = 0.9
results_path = './autoencoders/Results/Standard_AE'
saved_model_path = results_path + '/Saved_models/'

# Placeholders for input data and the targets
x_input = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='Input')
x_target = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='Target')
decoder_input = tf.placeholder(dtype=tf.float32, shape=[1, z_dim], name='Decoder_input')

recontruct = True

def dense(x, n1, n2, name):
    """
    Used to create a dense layer.
    :param x: input tensor to the dense layer
    :param n1: no. of input neurons
    :param n2: no. of output neurons
    :param name: name of the entire dense layer.i.e, variable scope name.
    :return: tensor with shape [batch_size, n2]
    """
    with tf.variable_scope(name, reuse=None):
        weights = tf.get_variable("weights", shape=[n1, n2],
                                  initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
        bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out


# The Encoder of the network
def encoder(x, reuse=False):
    """
    Encode part of the autoencoder
    :param x: input to the autoencoder
    :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
    :return: tensor which is the hidden latent variable of the autoencoder.
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Encoder'):
        e_dense_1 = tf.nn.tanh(dense(x, input_dim, hidden_size1, 'e_dense_1'))
        e_dense_2 = tf.nn.tanh(dense(e_dense_1, hidden_size1, hidden_size2, 'e_dense_2'))
        e_dense_3 = tf.nn.tanh(dense(e_dense_2, hidden_size2, hidden_size3, 'e_dense_3'))
        #latent_variable = dense(e_dense_3, hidden_size3, z_dim, 'e_latent_variable')
        latent_variable = tf.nn.tanh(dense(e_dense_3, hidden_size3, z_dim, 'e_latent_variable'))
        return latent_variable


# The Decoder of the network
def decoder(x, reuse=False):
    """
    Decoder part of the autoencoder
    :param x: input to the decoder
    :param reuse: True -> Reuse the decoder variables, False -> Create or search of variables before creating
    :return: tensor which should ideally be the input given to the encoder.
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Decoder'):
        d_dense_1 = tf.nn.tanh(dense(x, z_dim, hidden_size3, 'd_dense_1'))
        d_dense_2 = tf.nn.tanh(dense(d_dense_1, hidden_size3, hidden_size2, 'd_dense_2'))
        e_dense_3 = tf.nn.tanh(dense(d_dense_2, hidden_size2, hidden_size1, 'd_dense_3'))
        output = tf.nn.tanh(dense(e_dense_3, hidden_size1, input_dim, 'd_output'))
        return output


def reconstruct_variables(sess=None, op=None, data=None):
    # run the trained AE for predictions on the test data
    reconstructed_data = sess.run(op, feed_dict={x_input: data})
    print('Reconstructed data shape: {}'.format(reconstructed_data.shape))
    # We are going to plot the reconstructed data below


def train(train_model=True, train_data=None, test_data=None):
    """
    Used to train the autoencoder by passing in the necessary inputs.
    :param train_model: True -> Train the model, False -> Load the latest trained model and show the reconstructed variables.
    """
    with tf.variable_scope(tf.get_variable_scope()):
        encoder_output = encoder(x_input)
        decoder_output = decoder(encoder_output)

    with tf.variable_scope(tf.get_variable_scope()):
        reconstructed_variables = decoder(decoder_input, reuse=True)

    # Loss
    loss = tf.reduce_mean(tf.square(x_target - decoder_output))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(loss)
    init = tf.global_variables_initializer()

    # Saving the model
    saver = tf.train.Saver()
    step = 0
    with tf.Session() as sess:
        sess.run(init)
        if train_model:
            start = time.time()
            for i in range(n_epochs):
                train_data = shuffle(train_data)
                # break the train data df into chunks of size batch_size
                train_batches = [train_data[x:x + batch_size] for x in range(0, train_data.shape[0], batch_size)]

                mean_loss = 0.0
                for batch in train_batches:
                    sess.run(optimizer, feed_dict={x_input: batch, x_target: batch})

                    batch_loss = sess.run([loss], feed_dict={x_input: batch, x_target: batch})
                    mean_loss += batch_loss[0]
                    step += 1

                # Calculate the mean loss over all batches in one epoch
                mean_loss = float(mean_loss)/len(train_batches)
                # Saving takes a lot of time
                # saver.save(sess, save_path=saved_model_path, global_step=step)
                print("Model Trained!")

                validation_loss = sess.run([loss], feed_dict={x_input: test_data, x_target: test_data})
                print('\n-------------------------------------------------------------\n')
                print('Train loss after epoch {}: {}'.format(i, mean_loss))
                print('Validation loss after epoch {}: {}'.format(i, validation_loss))
                print("Elapsed time {:.2f} sec".format(time.time() - start))
                print('\n-------------------------------------------------------------\n')

            # print("Saved Model Path: {}".format(saved_model_path))

            if recontruct == True:
                reconstruct_variables(sess=sess, op=decoder_output, data=test_data)

        else:
            all_results = os.listdir(results_path)
            all_results.sort()
            saver.restore(sess,
                          save_path=tf.train.latest_checkpoint(results_path + '/' + all_results[-1] + '/Saved_models/'))
