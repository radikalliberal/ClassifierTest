import datetime
import glob
import os
from sys import platform

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

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

BATCH_SIZE = 50


def timestamp():
    return str(datetime.datetime.now()).split('.')[0]


def read_and_decode(fname_queue):
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
    label = tf.stack([label, tf.square(label - 1)], axis=0)
    # label = tf.concat([tf.expand_dims(label, 0), tf.square(tf.expand_dims(label, 0)-1)], 1)

    image_shape = [256, 256, 3]
    # label_shape = [2]

    image = tf.reshape(image, image_shape)
    # label = tf.reshape(label, label_shape)

    image = tf.cast(image, tf.float32) * (1 / 255) - 0.5

    # print(image)
    # image = tf.reshape(image, image_shape)
    # label = tf.reshape(label, label_shape)
    min_after_dequeue = BATCH_SIZE * 4
    capacity = min_after_dequeue + 400 * BATCH_SIZE

    images, labels = tf.train.shuffle_batch([image, label],
                                            batch_size=BATCH_SIZE,
                                            capacity=capacity,
                                            num_threads=2,
                                            min_after_dequeue=min_after_dequeue)
    return images, labels


def cnn_net(x, layers):
    out = x
    for i in range(layers):
        conv1 = tf.layers.conv2d(
            inputs=out,
            filters=16,
            kernel_size=[16, 16],
            strides=[4, 4],
            padding="same",
            activation=tf.nn.relu,
            trainable=True,
            use_bias=True,
            name=f'conv{i*2}')

        tf.summary.image(conv1.name, tf.expand_dims(conv1[:, :, :, 0], axis=3))

        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=16,
            kernel_size=[16, 16],
            strides=[2, 2],
            padding="same",
            activation=tf.nn.relu,
            trainable=True,
            use_bias=True,
            name=f'conv{i*2+1}')

        tf.summary.image(conv2.name, tf.expand_dims(conv2[:, :, :, 0], axis=3))

        print(conv1)
        print(conv2)

        pool = tf.layers.max_pooling2d(inputs=conv2,
                                       pool_size=[2, 2],
                                       strides=2,
                                       name=f'pool{i}')
        print(pool)

        out = pool
    return out


def main():
    ###################
    #### Build Net ####
    ###################
    with tf.device(DEV):
        filename_queue = tf.train.string_input_producer(
            glob.glob(f'{BASE_PATH}/ClassifierDataTfRecords/Train*.tfrecords'))

        imgs, lbls = read_and_decode(filename_queue)

        filename_queue = tf.train.string_input_producer(
            glob.glob(f'{BASE_PATH}/ClassifierDataTfRecords/Test*.tfrecords'))

        testimgs, testlbls = read_and_decode(filename_queue)

        x = tf.placeholder(tf.float32, [None, 256, 256, 3], name='Input')
        tf.summary.image(x.name, x)
        endpool = cnn_net(x, 2)

        print(endpool)

        pool3_flat = tf.reshape(endpool, [-1, 16])

        dropout = tf.layers.dropout(
            inputs=pool3_flat,
            rate=0.4,
            training=True)

        dense = tf.layers.dense(inputs=dropout,
                                units=2,
                                activation=tf.nn.relu,
                                use_bias=True,
                                name='logits')
        logits = tf.nn.softmax(dense, dim=1)
        # logits = dense

        # onehot_labels = tf.one_hot(indices=tf.cast(lbls_idx, tf.int32), depth=2)
        onehot_labels = tf.placeholder(tf.float32, [None, 2])
        print(onehot_labels)
        with tf.name_scope('cross_entropy'):
            diff = tf.nn.softmax_cross_entropy_with_logits(
                labels=onehot_labels,
                logits=logits)
            with tf.name_scope('total'):
                cross_entropy = tf.reduce_mean(diff)
                tf.summary.scalar('cross_entropy', cross_entropy)

        with tf.name_scope('train'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
            train_op = optimizer.minimize(loss=cross_entropy)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(onehot_labels, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        # y_ = tf.placeholder(tf.float32, [None, 2])

        img = (x * 256) + 128

        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)

        merged = tf.summary.merge_all()

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            writer = tf.summary.FileWriter(f'{BASE_PATH}/cnnlog', sess.graph)
            saver = tf.train.Saver()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            with open('./loss.log', '+w') as losslog:
                losslog.write(f'{timestamp()}:Started Learning')
                train_cumsum = np.array([0, 0])
                test_cumsum = np.array([0, 0])

                try:
                    for epoch in range(10):
                        # print(sess.run(imgs.eval()))
                        _ = sess.run([train_op],
                                     feed_dict={x: imgs.eval(),
                                                onehot_labels: lbls.eval()})

                        _, acc_t, los, y__ = sess.run([train_op, accuracy, cross_entropy, onehot_labels],
                                                      feed_dict={x: imgs.eval(),
                                                                 onehot_labels: lbls.eval()})
                        acc, y, y_, mrg = sess.run([accuracy, logits, onehot_labels, merged],
                                                   feed_dict={x: imgs.eval(),
                                                              onehot_labels: lbls.eval()})
                        # print(np.concatenate((out, truth), axis=1))
                        # print(np.concatenate((out, truth), axis=1))
                        # loss = sess.run(tf.reduce_sum(loss))
                        os.system(CLEAR)

                        train_cumsum = train_cumsum + np.sum(y__, axis=0)
                        test_cumsum = test_cumsum + np.sum(y_, axis=0)

                        writer.add_summary(mrg, epoch)
                        # writer.add_summary(tf.summary.scalar("loss", loss), i)
                        # writer.add_summary(tf.summary.scalar("Test_Accuracy", accuracy), i)
                        print(np.concatenate((y[:10], y_[:10]), axis=1))
                        print(f'max true:{max(y[:,0]):.4f}, max false:{max(y[:,1]):.4f}')
                        print(f'mean true:{np.mean(y[:,0]):.4f}, mean false:{np.mean(y[:,1]):.4f}')
                        print(f'min true:{min(y[:,0]):.4f}, min false:{min(y[:,1]):.4f}')
                        print(
                            f'spread face true:{max(y[:,0])-min(y[:,0]):.4f}, spread face false:{max(y[:,1])-min(y[:,1]):.4f}')
                        print(f'Train labels cum:{(train_cumsum/(epoch+1))/BATCH_SIZE}')
                        print(f'Test labels cum:{(test_cumsum/(epoch+1))/BATCH_SIZE}')
                        print(f'loss:{los:.6f}')
                        print(f'Train acc:{acc_t:.6f}')
                        print(f'Test acc:{acc:.6f}')
                        writer.flush()
                        if not epoch % 10:
                            saver.save(sess, f'{BASE_PATH}/cnnlog/')

                except tf.errors.OutOfRangeError as e:
                    print(e)

                finally:
                    writer.close()
                    coord.request_stop()
                    coord.join(threads)


if __name__ == '__main__':
    main()
