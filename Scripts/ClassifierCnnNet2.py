import glob
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


def read_and_decode(fname_queue, batch_size):
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
    min_after_dequeue = batch_size * 4
    capacity = min_after_dequeue + 400 * batch_size

    images, labels = tf.train.shuffle_batch([image, label],
                                            batch_size=batch_size,
                                            capacity=capacity,
                                            num_threads=2,
                                            min_after_dequeue=min_after_dequeue)
    return images, labels


def cnn_net(x, layers, act_func):
    out = x
    for i in range(layers):
        conv1 = tf.layers.conv2d(
            inputs=out,
            filters=16,
            kernel_size=[8, 8],
            strides=[1, 1],
            padding="same",
            activation=act_func,
            trainable=True,
            use_bias=True,
            name=f'conv{i*2}')

        tf.summary.image(conv1.name, tf.expand_dims(conv1[:, :, :, 0], axis=3), max_outputs=1)

        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=16,
            kernel_size=[8, 8],
            strides=[1, 1],
            padding="same",
            activation=act_func,
            trainable=True,
            use_bias=True,
            name=f'conv{i*2+1}')

        tf.summary.image(conv2.name, tf.expand_dims(conv2[:, :, :, 0], axis=3), max_outputs=1)

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
    Batchsizes = [100]
    learning_rates = [0.01, 0.005, 0.001, 0.0005]
    activation_funcs = [tf.nn.relu, tf.nn.sigmoid, tf.nn.tanh, tf.nn.softplus]
    Epochs = 10000

    for BATCH_SIZE in Batchsizes:
        for learning_rate in learning_rates:
            for activation_func in activation_funcs:
                tf.reset_default_graph()
                # Build Neural Net
                with tf.device(DEV):

                    with tf.name_scope('Data_queues'):
                        filename_queue = tf.train.string_input_producer(
                            glob.glob(f'{BASE_PATH}/ClassifierDataTfRecords/Train*.tfrecords'))

                        imgs, lbls = read_and_decode(filename_queue, BATCH_SIZE)

                        filename_queue = tf.train.string_input_producer(
                            glob.glob(f'{BASE_PATH}/ClassifierDataTfRecords/Test*.tfrecords'))

                        testimgs, testlbls = read_and_decode(filename_queue, BATCH_SIZE)

                    #img_initializer = tf.placeholder(dtype=imgs.dtype,
                    #                                 shape=imgs.shape)
                    #x = tf.Variable(img_initializer, trainable=False, collections=[])
                    x = tf.placeholder(tf.float32, [None, 256, 256, 3], name='Input')
                    tf.summary.image(x.name, x, max_outputs=1)
                    endpool = cnn_net(x, 2, activation_func)

                    pool3_flat = tf.reshape(endpool, [-1, 64 * 64 * 16])

                    dropout = tf.layers.dropout(
                        inputs=pool3_flat,
                        rate=0.4,
                        training=True)

                    print(dropout)

                    dense1 = tf.layers.dense(inputs=dropout,
                                             units=64 * 64,
                                             activation=activation_func,
                                             use_bias=True,
                                             name='pre_logits')

                    dense2 = tf.layers.dense(inputs=dense1,
                                             units=2,
                                             activation=activation_func,
                                             use_bias=True,
                                             name='logits')

                    logits = tf.nn.softmax(dense2, dim=1)

                    onehot_labels = tf.placeholder(tf.float32, [None, 2])
                    #x = tf.Variable(imgs, trainable=False, collections=[])
                    print(onehot_labels)
                    with tf.name_scope('cross_entropy'):
                        diff = tf.nn.softmax_cross_entropy_with_logits(
                            labels=onehot_labels,
                            logits=logits)
                        with tf.name_scope('total'):
                            cross_entropy = tf.reduce_mean(diff)
                            tf.summary.scalar('cross_entropy', cross_entropy)

                    with tf.name_scope('train'):
                        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
                        global_step = tf.Variable(0, name='global_step', trainable=False)
                        train_op = optimizer.minimize(loss=cross_entropy, global_step=global_step)

                    with tf.name_scope('accuracy'):
                        with tf.name_scope('correct_prediction'):
                            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(onehot_labels, 1))
                        with tf.name_scope('accuracy'):
                            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    tf.summary.scalar('accuracy', accuracy)

                    for var in tf.trainable_variables():
                        tf.summary.histogram(var.name, var)

                    merged = tf.summary.merge_all()

                    init_op = tf.group(tf.global_variables_initializer(),
                                       tf.local_variables_initializer())

                    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
                    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
                # Train
                with tf.Session(config=config).as_default() as sess:
                    session_path = f'{BASE_PATH}/cnnlog/afn={activation_func.__name__},bs={BATCH_SIZE},lr={learning_rate:.0e}'

                    sess.run(init_op)
                    writer = tf.summary.FileWriter(f'{session_path}', sess.graph)
                    print(glob.glob(f'{session_path}/model*'))
                    models = [x for x in glob.glob(f'{session_path}/model*') if '.meta' in x]
                    if len(models) > 0:
                        print(max(models))
                        saver = tf.train.import_meta_graph(max(models))
                        saver.restore(sess, tf.train.latest_checkpoint(session_path))
                        print(f'restored model {session_path} \nEpoch: {global_step.eval()}')
                    else:
                        print(f'created new model {session_path}')
                        saver = tf.train.Saver()
                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(coord=coord)
                    step = 0
                    try:
                        while step < Epochs:
                            print(f'step:{step}')
                            if not step % 100 or step == Epochs - 1:
                                print('saving model')
                                saver.save(sess,
                                           f'{session_path}/model-{step:06d}')

                            _, step = sess.run([train_op,
                                                global_step],
                                               feed_dict={x: imgs.eval(),
                                                          onehot_labels: lbls.eval()})

                            if not step % 50:
                                acc_t, los = sess.run([accuracy,
                                                       cross_entropy],
                                                      feed_dict={x: imgs.eval(),
                                                                 onehot_labels: lbls.eval()})

                                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                                run_metadata = tf.RunMetadata()

                                acc, y, y_, mrg = sess.run([accuracy,
                                                            logits,
                                                            onehot_labels,
                                                            merged],
                                                           feed_dict={x: testimgs.eval(),
                                                                      onehot_labels: testlbls.eval()},
                                                           options=run_options,
                                                           run_metadata=run_metadata)

                                # os.system(CLEAR)
                                writer.add_run_metadata(run_metadata, f'step:{step}')
                                writer.add_summary(mrg, step)

                                print(
                                    f'Batchsize:{BATCH_SIZE}, '
                                    f'activation_func:{activation_func.__name__}, '
                                    f'lr:{learning_rate}, Epoch:{step}')
                                print(np.concatenate((y[:10], y_[:10]), axis=1))
                                print(f'max  true:{max(y[:,0]):.4f}, '
                                      f'max  false:{max(y[:,1]):.4f}')
                                print(f'mean true:{np.mean(y[:,0]):.4f}, '
                                      f'mean false:{np.mean(y[:,1]):.4f}')
                                print(f'min  true:{min(y[:,0]):.4f}, '
                                      f'min  false:{min(y[:,1]):.4f}')
                                print(
                                    f'spread true:{max(y[:,0])-min(y[:,0]):.4f}, spread false:{max(y[:,1])-min(y[:,1]):.4f}')
                                print(f'loss:{los:.6f}')
                                print(f'Train acc:{acc_t:.2f}')
                                print(f'Test acc:{acc:.2f}')

                                writer.flush()

                    except tf.errors.OutOfRangeError as e:
                        print(e)

                    finally:
                        writer.close()
                        coord.request_stop()
                        coord.join(threads)


if __name__ == '__main__':
    main()
