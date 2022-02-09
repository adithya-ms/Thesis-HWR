import tensorflow as tf
import tensorflow.keras.backend as K
import pdb
import numpy as np

class WERMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the Word Piece Error Rate
    """
    def __init__(self, name='WER_metric', **kwargs):
        super(WERMetric, self).__init__(name=name, **kwargs)
        self.wer_accumulator = self.add_weight(name="total_wer", initializer="zeros")
        self.counter = self.add_weight(name="wer_count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        input_shape = K.shape(y_pred)
        input_length = tf.ones(shape=input_shape[0]) * K.cast(input_shape[1], 'float32')

        decode, log = K.ctc_decode(y_pred,
                                    input_length,
                                    greedy=True)
        decode[0] = tf.where(decode[0] == -1, np.int64(0) , decode[0])

        decode = K.ctc_label_dense_to_sparse(decode[0], K.cast(input_length, 'int32'))
        y_true_sparse = K.ctc_label_dense_to_sparse(y_true, K.cast(input_length, 'int32'))

        #decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))
        distance = tf.edit_distance(decode, y_true_sparse, normalize=True)
        #pdb.set_trace()
        self.wer_accumulator.assign_add(tf.reduce_sum(distance))
        self.counter.assign_add(tf.cast(len(y_true), 'float32'))

    def result(self):
        return tf.math.divide_no_nan(self.wer_accumulator, self.counter)

    def reset_states(self):
        self.wer_accumulator.assign(0.0)
        self.counter.assign(0.0)


class SERMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the Sentence Error Rate
    """
    def __init__(self, name='SER_metric', **kwargs):
        super(SERMetric, self).__init__(name=name, **kwargs)
        self.ser_accumulator = self.add_weight(name="total_ser", initializer="zeros")
        self.counter = self.add_weight(name="ser_count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        input_shape = K.shape(y_pred)
        input_length = tf.ones(shape=input_shape[0]) * K.cast(input_shape[1], 'float32')

        decode, log = K.ctc_decode(y_pred,
                                    input_length,
                                    greedy=True)
        decode[0] = tf.where(decode[0] == -1, np.int64(0), decode[0])
        decode = K.ctc_label_dense_to_sparse(decode[0], K.cast(input_length, 'int32'))
        y_true_sparse = K.ctc_label_dense_to_sparse(y_true, K.cast(input_length, 'int32'))

        #decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))
        distance = tf.edit_distance(decode, y_true_sparse, normalize=True)
        
        correct_words_amount = tf.reduce_sum(tf.cast(tf.not_equal(distance, 0), tf.float32))

        self.ser_accumulator.assign_add(correct_words_amount)
        self.counter.assign_add(tf.cast(len(y_true), 'float32'))

    def result(self):
        return tf.math.divide_no_nan(self.ser_accumulator, self.counter)

    def reset_states(self):
        self.ser_accumulator.assign(0.0)
        self.counter.assign(0.0)