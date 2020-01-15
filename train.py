import os
import numpy as np
import tensorflow as tf
from network import SurfaceNet
from losses import mmd_loss

import h5py
import math
from scipy import signal


class Trainer(object):

    def __init__(self, train_path, test_path,
                 num_classes=5, use_fine_grained=False,
                 regression_type='slack_l1'):

        self.train_set = h5py.File(train_path, 'r')
        print('num train: ', self.train_set['signal'].shape[0])

        self.test_set = h5py.File(test_path, 'r')
        print('num test: ', self.test_set['signal'].shape[0])

        self.num_classes = num_classes
        self.use_fine_grained = use_fine_grained
        self.regression_type = regression_type
        os.mkdir('./runs') if not os.path.exists('./runs') else None
        return

    def filter(self, y):
        b, a = signal.butter(8, 0.4, 'lowpass')
        return signal.filtfilt(b, a, y)

    def mmd_loss(self, output_tf, label_tf):
        label = tf.floor(label_tf)
        total_loss = None
        for cls in range(self.num_classes):
            mask = tf.equal(label, cls)
            masked_output = tf.reshape(
                tf.boolean_mask(output_tf, mask),
                [-1, 1])
            true_distribution = tf.random_uniform(
                shape=tf.shape(masked_output),
                minval=-0.5,
                maxval=0.5)
            masked_output = masked_output - (cls + 0.5)
            loss = mmd_loss(masked_output, true_distribution)
            if total_loss is None:
                total_loss = loss
            else:
                total_loss = total_loss + loss
        return total_loss

    def kurtosis_tf(self, x):
        x = x - tf.reduce_mean(x)
        square_std = tf.square(tf.reduce_mean(tf.square(x)))
        forth_order = tf.reduce_mean(tf.square(tf.square(x)))
        return forth_order / (1e-12 + square_std)

    def forth_order(self, x):
        x = x - tf.reduce_mean(x)
        return tf.reduce_mean(tf.square(tf.square(x)))

    def square_var(self, x):
        x = x - tf.reduce_mean(x)
        return tf.square(tf.reduce_mean(tf.square(x)))

    def kurtosis_loss(self, output_tf, label_tf):
        label = tf.floor(label_tf)
        total_loss = None
        for cls in range(self.num_classes):
            mask = tf.equal(label, cls)
            masked_output = tf.boolean_mask(output_tf, mask)
            true_distribution = tf.random_uniform(
                shape=tf.shape(masked_output),
                minval=0,
                maxval=1)
            loss = tf.abs(self.forth_order(
                masked_output
            ) * self.square_var(true_distribution) -
                self.forth_order(
                true_distribution
            ) * self.square_var(masked_output))
            if total_loss is None:
                total_loss = loss
            else:
                total_loss = total_loss + loss
        return total_loss

    def loss_wrapper(self, loss, scope):
        with tf.variable_scope(scope):
            sigma = tf.get_variable('sgm_loss', shape=(1,),
                                    dtype=tf.float32,
                                    initializer=tf.initializers.random_uniform(
                minval=0.2, maxval=1))
            factor = tf.div(1.0, 2.0 * sigma * sigma)
            loss = tf.add(tf.multiply(factor, loss),
                          tf.log(tf.nn.relu(sigma) + 1e-12))
            return loss

    def get_data(self, stage, batch_size):
        if stage == 'train':
            data = self.train_set
            indices = np.random.randint(0, data['signal'].shape[0], batch_size)
        elif stage == 'test':
            data = self.test_set
            batch_size = data['signal'].shape[0]
            indices = np.arange(data['signal'].shape[0])
        else:
            print('Invalid stage: {}'.format(stage))
        signal = np.zeros((batch_size, 2048, 1, 1), dtype=np.float32)
        label = np.zeros((batch_size), dtype=np.float32)
        for index in range(batch_size):
            sig = data['signal'][indices[index]]
            label[index] = data['period'][indices[index]]

            if stage == 'train' and np.random.uniform(size=1) < 0.5:
                signal[index, :, 0, 0] = np.flip(sig, axis=0)
            else:
                signal[index, :, 0, 0] = sig
        return signal, label

    def log_softmax_loss(self, output_tf, label_tf):
        return -tf.reduce_mean(tf.nn.log_softmax(output_tf)*label_tf)

    def regression_loss(self, output_tf, label_tf, alpha=0.0, beta=0.5):
        diff = tf.abs(output_tf - label_tf)
        diff_smooth_l1 = tf.sqrt(
            tf.square(tf.nn.relu(diff - alpha)) + beta ** 2) - beta
        return tf.reduce_mean(diff_smooth_l1)

    def immediate_threshold_loss(self, output_tf, label_tf):
        diff = tf.abs(output_tf - label_tf)
        loss = tf.nn.relu(diff - 0.5)
        return tf.reduce_mean(loss)

    def all_threshold_loss(self, output_tf, label_tf):
        diff = tf.abs(output_tf - label_tf)
        loss = tf.square(tf.nn.relu(diff - 0.5))
        return tf.reduce_mean(loss)

    def accuracy(self, output_np, label_np):
        return np.mean(np.argmax(output_np, axis=1
                                 ) == np.argmax(label_np, axis=1))

    def regression_error(self, output_np, label_np):
        error = np.mean(np.abs(output_np - label_np))
        return error

    def regression_mae(self, x, y):
        e = np.mean(np.abs(x - y))
        return e

    def regression_rmse(self, x, y):
        e = np.sqrt(np.mean(np.square(x - y)))
        return e

    def precision(self, x, y, thres=0.25):
        correct = np.abs(x - y) < thres
        return np.mean(correct)

    def regression_accuracy(self, output_np, label_np):
        target = np.argmax(label_np, axis=1)
        pred = np.around(output_np)
        return np.mean(target == pred)

    def fusion_matrix(self, output_np, label_np):
        argmax_output = np.argmax(output_np, axis=1)
        argmax_label = np.argmax(label_np, axis=1)
        fusion_matrix = np.zeros((5, 5), dtype=np.int32)
        for index in range(output_np.shape[0]):
            fusion_matrix[argmax_output[index], argmax_label[index]] += 1
        return fusion_matrix

    def get_kurtosis_weight(self, step, steps, max_weight=1.0):
        return max_weight

    def array_inversion(self, x):
        num = x.shape[0]
        max_comb = num * (num - 1) / 2

        def merge_count_split_inv(a, b):
            output = []
            count = 0
            for a_index, a_value in enumerate(a):
                while (b and a_value > b[0]):
                    left_in_a = len(a) - a_index
                    count += left_in_a
                    output.append(b.pop(0))
                output.append(a_value)
            output.extend(b)
            return output, count

        def sort_count_inv(arr, length):
            if length == 1:
                return arr, 0
            else:
                half_len = length // 2
                first_half = arr[:half_len]
                second_half = arr[half_len:]
                a_output, a_count = sort_count_inv(first_half, len(first_half))
                b_output, b_count = sort_count_inv(
                    second_half, len(second_half))
                c_output, c_count = merge_count_split_inv(a_output, b_output)
            return c_output, a_count + b_count + c_count

        _, num_inv = sort_count_inv(list(x), num)
        return num_inv * 1.0 / max_comb

    def train(self, batch_size, steps):
        signal_tf = tf.placeholder(shape=[None, 2048, 1, 1], dtype=tf.float32)
        label_tf = tf.placeholder(shape=[None], dtype=tf.float32)
        is_training = tf.placeholder(shape=(), dtype=tf.bool)
        learning_rate = tf.placeholder(shape=(), dtype=tf.float32)
        kurtosis_weight = tf.placeholder(shape=(), dtype=tf.float32)
        net = SurfaceNet(is_training)
        net.build(signal_tf)
        reg_loss = tf.reduce_mean(tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES)) * 0.1

        if self.use_fine_grained:
            regression_loss = self.regression_loss(
                net.output_regression, label_tf,
                alpha=0.0, beta=0.0)
            kurtosis_loss = tf.constant(0.0, dtype=tf.float32)
            mmd_distribution_loss = tf.constant(0.0, dtype=tf.float32)

        elif self.regression_type == 'slack_l1':
            regression_loss = self.regression_loss(net.output_regression,
                                                   tf.floor(label_tf) + 0.5,
                                                   alpha=0.2, beta=0.1)
            kurtosis_loss = self.kurtosis_loss(
                net.output_regression, label_tf) * 10.0
            mmd_distribution_loss = self.mmd_loss(
                net.output_regression, label_tf) * 0.03
        elif self.regression_type == 'l1':
            regression_loss = self.regression_loss(net.output_regression,
                                                   tf.floor(label_tf) + 0.5,
                                                   alpha=0.0, beta=0.0)
            kurtosis_loss = tf.constant(0.0, dtype=tf.float32)
            mmd_distribution_loss = tf.constant(0.0, dtype=tf.float32)
        elif self.regression_type == 'l2':
            regression_loss = tf.reduce_mean(
                tf.square(net.output_regression -
                          tf.floor(label_tf) - 0.5))
            kurtosis_loss = tf.constant(0.0, dtype=tf.float32)
            mmd_distribution_loss = tf.constant(0.0, dtype=tf.float32)
        elif self.regression_type == 'imm':
            regression_loss = self.immediate_threshold_loss(
                net.output_regression,
                tf.floor(label_tf) + 0.5)
            kurtosis_loss = tf.constant(0.0, dtype=tf.float32)
            mmd_distribution_loss = tf.constant(0.0, dtype=tf.float32)
        elif self.regression_type == 'all':
            regression_loss = self.all_threshold_loss(net.output_regression,
                                                      tf.floor(label_tf) + 0.5)
            kurtosis_loss = tf.constant(0.0, dtype=tf.float32)
            mmd_distribution_loss = tf.constant(0.0, dtype=tf.float32)

        elif self.regression_type == 'all_dr':
            regression_loss = self.all_threshold_loss(net.output_regression,
                                                      tf.floor(label_tf) + 0.5)
            kurtosis_loss = self.kurtosis_loss(
                net.output_regression, label_tf) * 10.0
            mmd_distribution_loss = self.mmd_loss(
                net.output_regression, label_tf) * 0.01

        else:
            raise NotImplementedError

        loss = reg_loss + \
            regression_loss + \
            (kurtosis_loss +
             mmd_distribution_loss) * kurtosis_weight

        train_op = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(loss)

        init_global = tf.global_variables_initializer()
        init_local = tf.local_variables_initializer()

        saver = tf.train.Saver(max_to_keep=2)
        learning_rate_basic = 1e-4
        learning_rate_curr = 1e-4
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        min_mae = 1.0
        min_rmse = 1.0
        max_prec = 0.0
        min_inv = 1.0

        with tf.Session(config=config) as sess:
            sess.run(tf.group(init_global, init_local))
            for step in range(steps):
                learning_rate_curr = learning_rate_basic / \
                    (1+np.exp((step - steps//2)/(steps//4)))
                signal_np_train, label_np_train = self.get_data(
                    'train', batch_size)
                feed_dict = {signal_tf: signal_np_train,
                             label_tf: label_np_train,
                             kurtosis_weight: self.get_kurtosis_weight(
                                 step, steps),
                             is_training: True,
                             learning_rate: learning_rate_curr}

                _, loss_np, regression_loss_np, kurtosis_loss_np, \
                    mmd_distribution_loss_np, output_reg_np_train = \
                    sess.run([train_op, loss, regression_loss,
                              kurtosis_loss, mmd_distribution_loss,
                              net.output_regression], feed_dict=feed_dict)

                if step % 10 == 0:
                    error_np = self.regression_error(
                        output_reg_np_train, label_np_train)
                    print('completed: %.2f%%  loss: %.2f  regression: %.2f  kurtosis: %.2f  distribution: %.2f  error: %.2f' %
                          (step*100.0/steps, loss_np,
                              regression_loss_np, kurtosis_loss_np,
                           mmd_distribution_loss_np, error_np))

                if step % 100 == 0:
                    signal_np_test, label_np_test = self.get_data(
                        'test', batch_size)
                    num_runs = int(
                        math.ceil(signal_np_test.shape[0] * 1.0 / batch_size))
                    output_regression_test = []
                    for i in range(num_runs):
                        feed_dict = {signal_tf: signal_np_test[i * batch_size: (i+1) * batch_size],
                                     label_tf: label_np_test[i * batch_size: (i+1) * batch_size],
                                     is_training: False}
                        [output_curr] = sess.run(
                            [net.output_regression], feed_dict=feed_dict)
                        output_regression_test.append(output_curr)
                    output_reg_np_test = np.concatenate(
                        output_regression_test, axis=0)
                    mae = self.regression_mae(
                        output_reg_np_test, label_np_test)
                    if min_mae > mae:
                        min_mae = mae

                    rmse = self.regression_rmse(
                        output_reg_np_test, label_np_test)
                    if min_rmse > rmse:
                        min_rmse = rmse

                    prec = self.precision(
                        output_reg_np_test, label_np_test, 0.5)
                    if prec > max_prec:
                        max_prec = prec

                    inv = self.array_inversion(output_reg_np_test)
                    if min_inv > inv:
                        min_inv = inv

                    print('----- step {} -----'.format(step))
                    print('test mae: %.4f, rmse: %.4f, prec: %.4f, inv: %.4f' %
                          (min_mae, min_rmse, max_prec, min_inv))
                    print('------------------')
                if (step + 1) % 3000 == 0 or (step + 1) == steps:
                    saver.save(
                        sess, './runs/model_step_{}.ckpt'.format(step+1))
        return


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    trainer = Trainer('./data/train.hdf5',
                      './data/test.hdf5',
                      num_classes=5,
                      use_fine_grained=False,
                      regression_type='all_dr')
    trainer.train(256, 120000)
