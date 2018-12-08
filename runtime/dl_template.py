import tensorflow as tf
import numpy
import shutil
import yaml
import os
import gzip


def extract_data(filename, num_images, image_size):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(image_size * image_size * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float)
        data = data.reshape(num_images, image_size * image_size)
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


def neural_net(X, layer, n_net, weights):
    oped_data = X
    op_name = {}
    for n_op, op in enumerate(layer.get('op')):
        if op not in op_name:
            op_name[op] = 0
        if op == "conv":
            out = tf.nn.conv2d(
                oped_data, weights["W_net%d_%s%d" % (n_net, op, op_name[op])], strides=[1, 1, 1, 1], padding='SAME') + weights["b_net%d_%s%d" % (n_net, op, op_name[op])]
            oped_data = eval(
                "tf.nn." + layer.get('active_function')[n_op] + "(out)")
            op_name[op] = op_name[op] + 1
        if op == "pool":
            k = layer.get("op_size")[n_op]
            oped_data = tf.nn.max_pool(oped_data, ksize=[1, k, k, 1], strides=[
                                        1, k, k, 1], padding='SAME')
        if op == "fc":
            print(tf.size(oped_data))
            if op_name[op] == 0:
                oped_data = tf.reshape(
                    oped_data, [-1, layer.get("op_size")[n_op][0]])
            out = tf.matmul(oped_data, weights["W_net%d_%s%d" % (
                n_net, op, op_name[op])]) + weights["b_net%d_%s%d" % (n_net, op, op_name[op])]
            oped_data = eval(
                "tf.nn." + layer.get('active_function')[n_op] + "(out)")
            op_name[op] = op_name[op] + 1
    return oped_data


def run_model(path, config):
    learning_rate = config.get('rate')
    training_epochs = config.get('epoch')
    display_step = config.get('display_step')
    batch_size = config.get('batch_size')

    model = config.get('model')
    placeholders = config.get('data').get('placeholder')
    train_data = []
    input_type = placeholders.get('input_type')
    if input_type == 'array':
        train_data.append(numpy.asarray(placeholders.get('in')))
        train_data.append(numpy.asarray(placeholders.get('out')))
    elif input_type == 'file':
        if placeholders.get('data_type') == 'image':
            image_size = placeholders.get('image_size')
            train_data.append(extract_data(
                path + '/' + placeholders.get('in'), 60000, image_size))
            train_data.append(one_hot(extract_labels(
                path + '/' + placeholders.get('out'), 60000)))
    X = tf.placeholder(tf.float32, [None, image_size * image_size])
    Y = tf.placeholder(tf.float32, [None, placeholders.get('onehot_size')])

    layers = config.get('layer')
    weights = {}
    var_name = []
    op_name = {}
    if model == 'cnn':
        if placeholders.get('data_type') == 'image':
            image_size = placeholders.get('image_size')
        for n_net, lay in enumerate(layers):
            op_sizes = lay.get('op_size')
            op_type = lay.get('op_type')
            if op_type != 'network' and op_type != None:
                continue
            for n_op, op in enumerate(lay.get('op')):
                if op == "pool":
                    continue
                if op not in op_name:
                    op_name[op] = 0
                var_name = "W_net%d_%s%d" % (n_net, op, op_name[op])
                weights[var_name] = tf.Variable(
                    tf.truncated_normal(op_sizes[n_op], stddev=0.1))
                var_name = "b_net%d_%s%d" % (n_net, op, op_name[op])
                weights[var_name] = tf.Variable(
                    tf.constant(0.1, shape=[op_sizes[n_op][-1]]))
                # noargcheck = weights[var_name]
                op_name[op] = op_name[op] + 1

    if model == 'mse':
        pred = tf.add(tf.multiply(X, weights[0]), weights[1])
        cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / \
            (2 * train_data[0].shape[0])
    elif model == 'cnn':
        net_sets = {}
        for n_net, layer in enumerate(layers):
            op_type = layer.get('op_type')
            op_id = layer.get('op_id')
            if op_id == None:
                op_id = 'output'
            op_input = layer.get('input')
            if op_input == None:
                op_input = 'init'
            net_sets['init'] = tf.reshape(X, [-1, image_size, image_size, 1])
            o1 = net_sets['init']
            if op_type == 'network' or op_type == None:
                if op_id == 'output':
                    output = neural_net(o1, layer, n_net, weights)
                else:
                    o1 = neural_net(net_sets[op_input], layer, n_net, weights)
            if op_type == "merge":
                # net_sets[op_id] = tf.add(net_sets[op_input[0]], net_sets[op_input[1]])
                o1 = tf.add(o1, net_sets[op_input[1]])

        # output = net_sets['output']
        cost = tf.reduce_mean(-tf.reduce_sum(Y *
                                             tf.log(output), reduction_indices=[1]))

    opter = config.get('optimizer')
    if opter == "gd":
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate).minimize(cost)
    elif opter == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(cost)
    pred = tf.argmax(output, 1)
    correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        i = 0
        for epoch in range(training_epochs):
            if i + batch_size <= 60000:
                batch_x = train_data[0][i:i + batch_size]
                batch_y = train_data[1][i:i + batch_size]
                i = i + batch_size
            else:
                j = (i + batch_size) % 60000
                batch_x = numpy.concatenate(
                    (train_data[0][i:], train_data[0][:j]), axis=0)
                batch_y = numpy.concatenate(
                    (train_data[1][i:], train_data[1][:j]), axis=0)
                i = j
            sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})

            if (epoch + 1) % display_step == 0:
                acc, c = sess.run([accuracy, cost], feed_dict={
                    X: batch_x, Y: batch_y})
                print("Epoch:", '%04d' % (epoch + 1), "cost=",
                      "{:.9f}".format(c), "accuracy=%f" % (acc))

        print("Optimization Finished!")
        tf.app.flags.DEFINE_integer('model_version', int(config.get('model_version')), 'version number of the model.')
        export_path = os.path.join(tf.compat.as_bytes(path + '/models/' + config.get('model_name')),
                                   tf.compat.as_bytes(str(tf.app.flags.FLAGS.model_version)))
        if os.path.exists(export_path):
            shutil.rmtree(export_path)

        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        tensor_info_x = tf.saved_model.utils.build_tensor_info(X)
        tensor_info_y = tf.saved_model.utils.build_tensor_info(output)
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'images': tensor_info_x},
                outputs={'scores': tensor_info_y},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    prediction_signature,
            },
            main_op=tf.tables_initializer(),
            strip_default_attrs=True)

        builder.save()
