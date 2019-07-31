#自动编码器
import tensorflow as tf
import numpy as py
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=False)

learning_rate = 0.001
training_epochs = 5
batch_size = 256
display_step = 1
examples_to_show = 10
n_input = 784
X = tf.placeholder("float",[None,n_input])

n_hidden_1 = 256
n_hidden_2 = 128

weights = {'encoder_h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
           'encoder_h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
           'decoder_h1':tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1])),
           'decoder_h2':tf.Variable(tf.random_normal([n_hidden_1,n_input]))
           }
biases = {'encoder_b1':tf.Variable(tf.random_normal([n_hidden_1])),
          'encoder_b2':tf.Variable(tf.random_normal([n_hidden_2])),
          'decoder_b1':tf.Variable(tf.random_normal([n_hidden_1])),
          'decoder_b2':tf.Variable(tf.random_normal([n_input]))
          }

def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_h1']),biases['encoder_b1']))

    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['encoder_h2']),biases['encoder_b2']))

    return layer_2

def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['decoder_h1']),biases['decoder_b1']))

    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['decoder_h2']),biases['decoder_b2']))

    return layer_2

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_pred = decoder_op
y_true = X

cost = tf.reduce_mean(tf.pow(y_true-y_pred,2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            _,c = sess.run([optimizer,cost],feed_dict={X:batch_xs})

            if epoch % display_step ==0:
                print("epoth:",'%04d'%(epoch+1),"cost=","{:.9f}".format(c))
    print("optimiztion finished!")


#import tensorflow as tf
# Creates a graph.
# with tf.device('/gpu:0'):
"""a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))"""
