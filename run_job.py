from __future__ import print_function

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import yaml
import gzip
import sys, getopt
rng = numpy.random

path = sys.argv[1]
yaml_file = open(path + '/JOBCONFIG.yaml', 'r')
job_config = yaml.load(yaml_file)

print(job_config.get('type'))
config = job_config.get('job-config')

# Parameters
learning_rate = config.get('rate')
training_epochs = config.get('epoch')
display_step = config.get('display_step')
batch_size = config.get('batch_size')

def extract_data(filename, num_images, image_size):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(image_size*image_size*num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float)
        data = data.reshape(num_images, image_size*image_size)
        return data

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
        return labels

def one_hot(labels):
    n_sample = len(labels)
    one_hot_labels = numpy.zeros((n_sample, 10)).astype(numpy.int64)
    for ii in range(n_sample):
        one_hot_labels[ii][labels[ii]] += 1
    return one_hot_labels

model = config.get('model')
placeholders = config.get('data').get('placeholder')
# Training Data
train_data = []
train_holder = []
# for var in placeholders:
#     train_data.append(numpy.asarray(placeholders.get(var)))
    # train_holder.append(tf.placeholder("float"))
input_type = placeholders.get('input_type')
if input_type == 'array':
    train_data.append(numpy.asarray(placeholders.get('in')))
    train_data.append(numpy.asarray(placeholders.get('out')))
elif input_type == 'file':
    if placeholders.get('data_type') == 'image':
        image_size = placeholders.get('image_size')
        train_data.append(extract_data(path + '/' + placeholders.get('in'), 3000, image_size))
        train_data.append(one_hot(extract_labels(path + '/' + placeholders.get('out'), 3000)))
X = tf.placeholder("float", [None, image_size * image_size])
Y = tf.placeholder("float", [None, placeholders.get('onehot_size')])

variables = config.get('data').get('variable')
layers = config.get('layer')
weights = {}
ops = {}
for var in variables:
    var_name, op = var.items()[0]
    weights[var_name] = []
    ops[var_name] = op
if model == 'nn':
    if placeholders.get('data_type') == 'image':
        image_size = placeholders.get('image_size')
        former_size = image_size * image_size
    for lay in layers:
        for sz in lay.get('size'):
            for var in lay.get('variable'):
                varn = var['name']
                vard = var['dimension']
                if vard == 1:
                    dim = str(sz)
                else:
                    dim = "%d, %d" % (former_size, sz)
                weights[varn].append(tf.Variable(eval("%s([%s])" % (ops[varn], dim))))
            former_size = sz

def neural_net(x):
    oped_data = X
    for layer in layers:
        for i in range(layer.get('repeat')):
            for j, op in enumerate(layer.get('operation')):
                oped_data = eval(op + '(oped_data, weights[layer.get("variable")[j]["name"]][i])')
    return oped_data

if model == 'mse':
    pred = tf.add(tf.multiply(X, weights[0]), weights[1])
    cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * train_data[0].shape[0])
elif model == 'nn':
    logits = neural_net(X)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=Y))

opter = config.get('optimizer')
if opter == "gd":
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
elif opter == 'adam':
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    sess.run(init)
    i=0
    for epoch in range(training_epochs):
        if i+batch_size <= 3000:
            batch_x = train_data[0][i:i+batch_size]
            batch_y = train_data[1][i:i+batch_size]
            i = i+batch_size
        else:
            j = (i+batch_size)%3000
            batch_x = numpy.concatenate((train_data[0][i:], train_data[0][:j]), axis=0)
            batch_y = numpy.concatenate((train_data[1][i:], train_data[1][:j]), axis=0)
            i = j
        # for (x, y) in zip(train_data[0], train_data[1]):
        sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})

        if (epoch + 1) % display_step == 0:
            acc, c = sess.run([accuracy, cost], feed_dict={X: train_data[0], Y: train_data[1]})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), "accuracy=%f" % (acc))

    print("Optimization Finished!")
    acc, training_cost = sess.run([accuracy, cost], feed_dict={X: train_data[0], Y: train_data[1]})
    print("Training accuracy=", acc, '\n')


f = open(path + "/output.txt", "w")
f.write("succeed!!")
f.close()