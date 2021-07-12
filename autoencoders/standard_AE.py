import pandas as pd
import tensorflow as tf
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


class Standard_Autoencoder:
    def __init__(self, input_dim, z_dim):
        # Parameters
        self.input_dim = input_dim
        self.hidden_size1 = 100
        self.hidden_size2 = 200
        self.hidden_size3 = 100
        self.hidden_size4 = 50
        self.z_dim = z_dim

        self.batch_size = 256
        self.n_epochs = 30
        self.learning_rate = 0.001
        self.beta1 = 0.9
        self.results_path = './autoencoders/Results/Standard_AE'
        self.saved_model_path = self.results_path + '/Saved_models/'

        # Placeholders for input data and the targets
        self.x_input = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='Input')
        self.x_target = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='Target')
        self.decoder_input = tf.placeholder(dtype=tf.float32, shape=[1, z_dim], name='Decoder_input')

        self.recontruct = True

    def dense(self, x, n1, n2, name):
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
    def encoder(self, x, reuse=False):
        """
        Encode part of the autoencoder
        :param x: input to the autoencoder
        :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
        :return: tensor which is the hidden latent variable of the autoencoder.
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()
        with tf.name_scope('Encoder'):
            e_dense_1 = tf.nn.leaky_relu(self.dense(x, self.input_dim, self.hidden_size1, 'e_dense_1'))
            e_dense_2 = tf.nn.leaky_relu(self.dense(e_dense_1, self.hidden_size1, self.hidden_size2, 'e_dense_2'))
            e_dense_3 = tf.nn.leaky_relu(self.dense(e_dense_2, self.hidden_size2, self.hidden_size3, 'e_dense_3'))
            e_dense_4 = tf.nn.leaky_relu(self.dense(e_dense_3, self.hidden_size3, self.hidden_size4, 'e_dense_4'))
            # latent_variable = dense(e_dense_3, hidden_size3, z_dim, 'e_latent_variable')
            latent_variable = tf.nn.leaky_relu(self.dense(e_dense_4, self.hidden_size4, self.z_dim, 'e_latent_variable'))
            return latent_variable

    # The Decoder of the network
    def decoder(self, x, reuse=False):
        """
        Decoder part of the autoencoder
        :param x: input to the decoder
        :param reuse: True -> Reuse the decoder variables, False -> Create or search of variables before creating
        :return: tensor which should ideally be the input given to the encoder.
        """
        if reuse:
            tf.get_variable_scope().reuse_variables()
        with tf.name_scope('Decoder'):
            d_dense_1 = tf.nn.leaky_relu(self.dense(x, self.z_dim, self.hidden_size4, 'd_dense_1'))
            d_dense_2 = tf.nn.leaky_relu(self.dense(d_dense_1, self.hidden_size4, self.hidden_size3, 'd_dense_2'))
            d_dense_3 = tf.nn.leaky_relu(self.dense(d_dense_2, self.hidden_size3, self.hidden_size2, 'd_dense_3'))
            d_dense_4 = tf.nn.leaky_relu(self.dense(d_dense_3, self.hidden_size2, self.hidden_size1, 'd_dense_4'))
            output = self.dense(d_dense_4, self.hidden_size1, self.input_dim, 'd_output')
            return output

    def reconstruct_variables(self, sess=None, op=None, data=None):
        # run the trained AE for predictions on the test data
        reconstructed_data = sess.run(op, feed_dict={self.x_input: data})
        print('Reconstructed data shape: {}'.format(reconstructed_data.shape))
        return reconstructed_data

    def train(self, train_model=True, train_data=None, test_data=None):
        """
        Used to train the autoencoder by passing in the necessary inputs.
        :param train_model: True -> Train the model, False -> Load the latest trained model and show the reconstructed variables.
        """
        with tf.variable_scope(tf.get_variable_scope()):
            encoder_output = self.encoder(self.x_input)
            decoder_output = self.decoder(encoder_output)

        with tf.variable_scope(tf.get_variable_scope()):
            reconstructed_variables = self.decoder(self.decoder_input, reuse=True)

        # Loss
        loss = tf.reduce_mean(tf.square(self.x_target - decoder_output))

        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1).minimize(loss)
        init = tf.global_variables_initializer()

        # Saving the model
        saver = tf.train.Saver()
        step = 0
        with tf.Session() as sess:
            sess.run(init)
            if train_model:
                start = time.time()
                train_loss = []
                val_loss = []
                epochs = []
                for i in range(self.n_epochs):
                    epochs.append(i + 1)
                    train_data = shuffle(train_data)
                    # break the train data df into chunks of size batch_size
                    train_batches = [train_data[x:x + self.batch_size] for x in
                                     range(0, train_data.shape[0], self.batch_size)]

                    mean_loss = 0.0
                    for batch in train_batches:
                        sess.run(optimizer, feed_dict={self.x_input: batch, self.x_target: batch})

                        batch_loss = sess.run([loss], feed_dict={self.x_input: batch, self.x_target: batch})
                        mean_loss += batch_loss[0]
                        step += 1

                    # Calculate the mean loss over all batches in one epoch
                    mean_loss = float(mean_loss) / len(train_batches)

                    # store train loss for plotting
                    train_loss.append(mean_loss)
                    # Saving takes a lot of time
                    # saver.save(sess, save_path=saved_model_path, global_step=step)
                    print("Model Trained!")

                    validation_loss = sess.run([loss], feed_dict={self.x_input: test_data, self.x_target: test_data})
                    # store validation loss for plotting
                    val_loss.append(validation_loss)
                    print('\n-------------------------------------------------------------\n')
                    print('Train loss after epoch {}: {}'.format(i, mean_loss))
                    print('Validation loss after epoch {}: {}'.format(i, validation_loss))
                    print("Elapsed time {:.2f} sec".format(time.time() - start))
                    print('\n-------------------------------------------------------------\n')

                # print("Saved Model Path: {}".format(saved_model_path))
                plt.figure()
                plt.plot(epochs, train_loss, 'g-', label="Train_loss")
                plt.plot(epochs, val_loss, 'r-', label="Validation_loss")
                plt.title('AE loss vs epochs')
                plt.xlabel('Epochs')
                plt.ylabel('AE Loss')
                plt.xticks(epochs[0:19])
                plt.legend()
                plt.show()

                if self.recontruct:
                    reconstructed_data = self.reconstruct_variables(sess=sess, op=decoder_output, data=test_data)
                    return reconstructed_data

            else:
                all_results = os.listdir(self.results_path)
                all_results.sort()
                saver.restore(sess,
                              save_path=tf.train.latest_checkpoint(
                                  self.results_path + '/' + all_results[-1] + '/Saved_models/'))
