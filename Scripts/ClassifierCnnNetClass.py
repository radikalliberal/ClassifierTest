import glob
from sys import platform

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


class model:
    def __init__(self, batchsize=100, learn_rate=0.001, activation_func=tf.nn.relu, train=True, restore=True,
                 device='/gpu:0'):
        tf.reset_default_graph()
        self.restore_model = restore
        self.train = train
        self.batchsize = batchsize
        self.learn_rate = learn_rate
        self.activation_func = activation_func
        self.dev = device
        self.session_path = f'{BASE_PATH}/cnnlog/afn={activation_func.__name__},' \
                            f'bs={batchsize},' \
                            f'lr={learn_rate:.0e}'
        self.graph = tf.get_default_graph()
        self.imgs, self.lbls = self.build_queue()
        self.merged = tf.summary.merge_all()

        if not restore:
            with tf.device(self.dev):
                self.tensors, self.train_op = self.build_model()
                self.graph = tf.get_default_graph()
                self.saver = tf.train.Saver()
                self.create_summarys()

    def start_session(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        self.graph.as_default()
        return tf.Session(config=config, graph=self.graph)

    def build_queue(self):
        with tf.name_scope('Data_queues'):
            with tf.name_scope('Train'):
                filename_queue1 = tf.train.string_input_producer(
                    glob.glob(f'{BASE_PATH}/ClassifierDataTfRecords/Train*.tfrecords'))

            with tf.name_scope('Test'):
                filename_queue2 = tf.train.string_input_producer(
                    glob.glob(f'{BASE_PATH}/ClassifierDataTfRecords/Test*.tfrecords'))

        if self.train:
            return read_and_decode(filename_queue1, self.batchsize)
        else:
            return read_and_decode(filename_queue2, self.batchsize)

    def create_summarys(self):
        with tf.name_scope('images'):
            tf.summary.image(self.imgs.name, self.imgs, max_outputs=1)
            for tensor in tf.all_variables():
                if 'conv' in tensor.name:
                    pass
                    # tf.summary.image(tensor.name, tf.expand_dims(tensor[:, :, :, 0], axis=3), max_outputs=1)

        with tf.name_scope('scalars'):
            for tensor in tf.all_variables():
                if 'cross_entropy/cross_entropy' in tensor.name or 'accuracy' in tensor.name:
                    pass
                    # tf.summary.scalar(tensor.name, tensor)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)

    def build_model(self):
        out = self.imgs
        Tensors = {}

        with tf.name_scope('dnn'):
            for i in range(2):
                conv1 = tf.layers.conv2d(
                    inputs=out,
                    filters=16,
                    kernel_size=[8, 8],
                    strides=[1, 1],
                    padding="same",
                    activation=self.activation_func,
                    trainable=True,
                    use_bias=True,
                    name=f'conv{i*2}')

                Tensors[conv1.name] = conv1
                # tf.summary.image(conv1.name, tf.expand_dims(conv1[:, :, :, 0], axis=3), max_outputs=1)

                conv2 = tf.layers.conv2d(
                    inputs=conv1,
                    filters=16,
                    kernel_size=[8, 8],
                    strides=[1, 1],
                    padding="same",
                    activation=self.activation_func,
                    trainable=True,
                    use_bias=True,
                    name=f'conv{i*2+1}')

                Tensors[conv2.name] = conv2
                # tf.summary.image(conv2.name, tf.expand_dims(conv2[:, :, :, 0], axis=3), max_outputs=1)

                pool = tf.layers.max_pooling2d(inputs=conv2,
                                               pool_size=[2, 2],
                                               strides=2,
                                               name=f'pool{i}')
                Tensors[pool.name] = pool

                out = pool

            pool3_flat = tf.reshape(out, [-1, 64 * 64 * 16])
            Tensors[pool3_flat.name] = pool3_flat

            dense1 = tf.layers.dense(inputs=pool3_flat,
                                     units=64 * 64,
                                     activation=self.activation_func,
                                     use_bias=True,
                                     name='pre_logits')

            dropout = tf.layers.dropout(
                inputs=dense1,
                rate=0.4,
                training=True)

            Tensors[dropout.name] = dropout

            dense2 = tf.layers.dense(inputs=dropout,
                                     units=2,
                                     activation=self.activation_func,
                                     use_bias=True,
                                     name='logits')

            logits = tf.nn.softmax(dense2, dim=1)

            Tensors[dense1.name] = dense1
            Tensors[dense2.name] = dense2
            Tensors[logits.name] = logits
            with tf.name_scope('cross_entropy'):
                diff = tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.lbls,
                    logits=logits,
                    name='diff')
                cross_entropy = tf.reduce_mean(diff, name='cross_entropy')

            Tensors[diff.name] = diff
            Tensors[cross_entropy.name] = cross_entropy

        with tf.name_scope('train'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learn_rate)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_op = optimizer.minimize(loss=cross_entropy, global_step=global_step, name='train_op')
            Tensors[global_step.name] = global_step

        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.lbls, 1))
            Tensors[correct_prediction.name] = correct_prediction
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            Tensors[accuracy.name] = accuracy

        return Tensors, train_op

    def __enter__(self):
        pass

    def __exit__(self):
        pass

    def train(self):
        pass

    def save(self):
        pass

    def tensor(self, name):
        return self.graph.get_tensor_by_name(name)

    def operation(self, name):
        return self.graph.get_operation_by_name(name)

    def restore(self, session):
        if self.restore_model:

            models = [x for x in glob.glob(f'{self.session_path}/model*') if '.meta' in x]
            if len(models) > 0:
                print(max(models))
                self.saver = tf.train.import_meta_graph(max(models))

            # tf.reset_default_graph()
            path = self.saver.restore(session, tf.train.latest_checkpoint(self.session_path))
            self.graph = tf.get_default_graph()
            # for key, tensor in self.tensors.items():
            #    self.tensors[key] = self.graph.get_tensor_by_name(tensor.name)

            step = self.graph.get_tensor_by_name('train/global_step:0')

            print(f'restored model {tf.train.latest_checkpoint(self.session_path)} Epoch: {step.eval(session=session)}')

            # self.global_step = self.graph.get_tensor_by_name(self.global_step.name)
            self.train_op = self.operation('train/train_op')
            # self.create_summarys()


def main():
    Batchsizes = [100]
    learning_rates = [0.01, 0.005, 0.001, 0.0005]
    activation_funcs = [tf.nn.relu, tf.nn.sigmoid, tf.nn.tanh, tf.nn.softplus]
    Epochs = 5000
    Test = False

    # Train / Test

    mod = model(restore=False, train=True)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # with tf.Session(config=config).as_default() as sess:
    with mod.start_session() as sess:
        sess.run(init_op)
        mod.restore(sess)

        mod.writer = tf.summary.FileWriter(f'{mod.session_path}', mod.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        step = 0
        try:
            while step < Epochs:
                if (not step % 100 or step == Epochs - 1) and not Test:
                    print('saving model')
                    mod.saver.save(sess,
                                   f'{mod.session_path}/model-{step:06d}')
                if mod.train:
                    _, step, los = sess.run([mod.train_op,
                                             mod.tensor('train/global_step:0'),
                                             mod.tensor('dnn/cross_entropy/cross_entropy:0')])

                    print(f'Epoch:{step}, loss:{los:.6f}')

                if not step % 10:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
                    run_metadata = tf.RunMetadata()

                    acc, los, mrg = sess.run([mod.tensor('accuracy/Mean:0'),
                                              mod.tensor('dnn/cross_entropy/cross_entropy:0'),
                                              mod.merged],
                                             options=run_options,
                                             run_metadata=run_metadata)

                    # os.system(CLEAR)
                    mod.writer.add_run_metadata(run_metadata, f'step:{step}')
                    mod.writer.add_summary(mrg, step)

                    print(
                        f'Epoch:{step},'
                        f'Batchsize:{mod.batchsize}, '
                        f'activation_func:{mod.activation_func.__name__}, '
                        f'lr:{mod.learn_rate}')
                    print(f'loss:{los:.6f}')
                    print(f'acc:{acc:.2f}')

                    mod.writer.flush()

        except tf.errors.OutOfRangeError as e:
            print(e)

        finally:
            mod.writer.close()
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    main()
