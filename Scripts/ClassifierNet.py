import glob
import os
from sys import platform

import numpy as np
import tensorflow as tf

if platform == 'win32':
    BASE_PATH = 'D:/Temp'
    DATASET_PATH = 'D:/Temp/ClassifierData'
    CLEAR = 'cls'
    DEV = '/gpu:0'
elif platform == 'linux':
    BASE_PATH = '/home/jscholz/Code/ClassifierTest'
    DATASET_PATH = '/home/jscholz/Code/ClassifierTest/Dataset'
    CLEAR = 'clear'
    DEV = '/gpu:0'


def read_and_decode(fname_queue, batchsize):
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(fname_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        })

    image = tf.decode_raw(features['image'], tf.uint8)

    # label = features['label']
    # print(label)

    label = tf.cast(features['label'], tf.int64)
    # t1 = tf.constant(features['label'])
    label = tf.stack([label, tf.square(label - 1)], axis=0)
    # label = tf.concat([tf.expand_dims(label, 0), tf.square(tf.expand_dims(label, 0)-1)], 1)

    image_shape = [256 * 256 * 3]
    # label_shape = [2]

    image = tf.reshape(image, image_shape)
    # label = tf.reshape(label, label_shape)

    image = tf.cast(image, tf.float32) * (1 / 255) - 0.5

    print(image)
    # image = tf.reshape(image, image_shape)
    # label = tf.reshape(label, label_shape)
    min_after_dequeue = 400
    capacity = min_after_dequeue + 3 * batchsize

    images, labels = tf.train.shuffle_batch([image, label],
                                            batch_size=batchsize,
                                            capacity=capacity,
                                            num_threads=2,
                                            min_after_dequeue=min_after_dequeue)
    return images, labels


def layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 0.1
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="weights")
        b = tf.Variable(tf.zeros([n_neurons]), name='biases')
        z = tf.matmul(X, W) + b
        if activation is None:
            return z
        else:
            return activation(z)


n_inputs = 256 * 256 * 3
# n_hidden1 = 256*32*3
n_hidden2 = 256 * 4 * 3
n_hidden3 = 32 * 4 * 3
n_hidden4 = 4 * 4 * 3
n_hidden5 = 2 * 3
n_outputs = 2

# BATCH_SIZE = 100
# learning_rate = 0.01
# activation_func = tf.nn.tanh

Batchsizes = [100, 500]
learning_rates = [0.005, 0.001, 0.0005]
activation_funcs = [tf.nn.relu, tf.nn.sigmoid, tf.nn.tanh, tf.nn.softplus]

for BATCH_SIZE in Batchsizes:
    for learning_rate in learning_rates:
        for activation_func in activation_funcs:
            tf.reset_default_graph()

            filename_queue1 = tf.train.string_input_producer(
                glob.glob(f'{BASE_PATH}/ClassifierDataTfRecords/Train*.tfrecords'))
            filename_queue2 = tf.train.string_input_producer(
                glob.glob(f'{BASE_PATH}/ClassifierDataTfRecords/Test*.tfrecords'))

            imgs, lbls = read_and_decode(filename_queue1, BATCH_SIZE)
            testimgs, testlbls = read_and_decode(filename_queue2, BATCH_SIZE)

            with tf.device(DEV):
                x = tf.placeholder(tf.float32, [BATCH_SIZE, 256 * 256 * 3])
                y_ = tf.placeholder(tf.float32, [BATCH_SIZE, 2])
                with tf.name_scope('dnn'):
                    # hidden1 = layer(X, n_hidden1, 'hidden1', activation='relu')
                    hidden2 = layer(x, n_hidden2, 'hidden2', activation=activation_func)
                    hidden3 = layer(hidden2, n_hidden3, 'hidden3', activation=activation_func)
                    hidden4 = layer(hidden3, n_hidden4, 'hidden4', activation=activation_func)
                    hidden5 = layer(hidden4, n_hidden5, 'hidden5', activation=activation_func)
                    y = layer(hidden5, n_outputs, 'outputs', activation=tf.nn.softmax)

                with tf.name_scope("cost"):
                    tau = 0.1
                    face, no_face = tf.split(y, num_or_size_splits=2, axis=1)
                    # spread_maximisation = tau/(tf.reduce_max(face)-tf.reduce_min(face) + tf.reduce_max(no_face) - tf.reduce_min(no_face)) -0.5
                    face_spread = tf.contrib.distributions.percentile(face, q=80) - tf.contrib.distributions.percentile(
                        face, q=20)
                    no_face_spread = tf.contrib.distributions.percentile(no_face,
                                                                         q=80) - tf.contrib.distributions.percentile(
                        no_face, q=20)
                    spread_maximisation = tau / (face_spread + no_face_spread) - tau / 2
                    cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_, predictions=y)) + spread_maximisation
                    tf.summary.scalar('cost', cost)

                with tf.name_scope('gradients'):
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                    # Op to calculate every variable gradient
                    grads = tf.gradients(cost, tf.trainable_variables())
                    grads = list(zip(grads, tf.trainable_variables()))
                    # Op to update all variables according to their gradient
                    train_step = optimizer.apply_gradients(grads_and_vars=grads)
                    # train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
                with tf.name_scope("accuracy"):
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    tf.summary.scalar('accuray', accuracy)

                for var in tf.trainable_variables():
                    tf.summary.histogram(var.name, var)
                # Summarize all gradients
                for grad, var in grads:
                    if grad is None:
                        continue
                    tf.summary.histogram(var.name + '/gradient', grad)

                init_op = tf.group(tf.global_variables_initializer(),
                                   tf.local_variables_initializer())
                config = tf.ConfigProto(allow_soft_placement=True)
            with tf.Session(config=config) as sess:
                sess.run(init_op)

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
                writer = tf.summary.FileWriter(
                    f'{BASE_PATH}/log/{activation_func.__name__}_{BATCH_SIZE}_{learning_rate:.0e}', sess.graph)
                merged = tf.summary.merge_all()
                saver = tf.train.Saver()
                try:
                    for epoch in range(500):
                        # print(sess.run(imgs.eval()))

                        _, loss, out, truth = sess.run([train_step, cost, y, y_],
                                                       feed_dict={x: imgs.eval(session=sess),
                                                                  y_: lbls.eval(session=sess)})
                        # print(np.concatenate((out, truth), axis=1))

                        if not epoch % 10:
                            acc, summary = sess.run([accuracy, merged],
                                                    feed_dict={x: testimgs.eval(session=sess),
                                                               y_: testlbls.eval(session=sess)})

                            writer.add_summary(summary, epoch)
                            # print(np.concatenate((out[:10], truth[:10]), axis=1))
                            os.system(CLEAR)
                            print(
                                f'Batchsize:{BATCH_SIZE}, activation_func:{activation_func.__name__}, lr:{learning_rate}, Epoch:{epoch}')
                            print(f'max true:{max(out[:,0]):.4f}, max false:{max(out[:,1]):.4f}')
                            print(f'mean true:{np.mean(out[:,0]):.4f}, mean false:{np.mean(out[:,1]):.4f}')
                            print(f'min true:{min(out[:,0]):.4f}, min false:{min(out[:,1]):.4f}')
                            print(
                                f'spread face true:{max(out[:,0])-min(out[:,0]):.4f}, spread face false:{max(out[:,1])-min(out[:,1]):.4f}')
                            loss = sess.run(tf.reduce_sum(loss))
                            print(f'acc:{acc:.4f}, loss:{loss:.4f}')
                            writer.flush()
                    if not epoch % 100:
                        saver.save(sess, f'{BASE_PATH}/log/{activation_func.__name__}_{BATCH_SIZE}_{learning_rate:.0e}')
                except (tf.errors.OutOfRangeError, tf.errors.InvalidArgumentError) as e:
                    print(e)

                finally:
                    coord.request_stop()
                    coord.join(threads)
