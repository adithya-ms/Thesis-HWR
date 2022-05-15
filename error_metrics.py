import tensorflow as tf
import tensorflow.keras.backend as K
import pdb
import numpy as np
from Levenshtein import distance as levenshtein_distance

class CERMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the Character Error Rate
    """
    def __init__(self, name='CER_metric', **kwargs):
        super(CERMetric, self).__init__(name=name, **kwargs)
        self.cer_accumulator = self.add_weight(name="total_wer", initializer="zeros")
        self.counter = self.add_weight(name="cer_count", initializer="zeros")

    def update_state(self, y_true, y_pred, tokenizer):
        input_shape = K.shape(y_pred)
        input_length = tf.ones(shape=input_shape[0]) * K.cast(input_shape[1], 'float32')

        decode, log = K.ctc_decode(y_pred,
                                    input_length,
                                    greedy=True)
        decode[0] = tf.where(decode[0] == -1, np.int64(0) , decode[0])

        decode = K.ctc_label_dense_to_sparse(decode[0], K.cast(input_length, 'int32'))
        #y_true_sparse = K.ctc_label_dense_to_sparse(y_true, K.cast(input_length, 'int32'))

        #decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))
        decode = tf.cast(decode, 'int64')
        y_true = tf.cast(y_true, 'int64')
        
        decode = tf.sparse.to_dense(decode)

        predicted_text = tokenizer.en.detokenize(decode)
        tar_text = tokenizer.en.detokenize(y_true)
        cer = []
        for index in range(0,predicted_text.shape[0]):
            cer.append(float(levenshtein_distance(predicted_text[index].numpy(), tar_text[index].numpy())))
        #pdb.set_trace()
        
        self.cer_accumulator.assign_add(tf.reduce_sum(cer))
        self.counter.assign_add(tf.cast(len(y_true), 'float32'))

    def result(self):
        return tf.math.divide_no_nan(self.cer_accumulator, self.counter)

    def reset_states(self):
        self.cer_accumulator.assign(0.0)
        self.counter.assign(0.0)


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
        decode = tf.cast(decode, 'int32')
        y_true_sparse = tf.cast(y_true_sparse, 'int32')
        distance = tf.edit_distance(decode, y_true_sparse, normalize=False)
        
        self.wer_accumulator.assign_add(tf.reduce_sum(distance))
        self.counter.assign_add(tf.cast(len(y_true), 'float32'))

    def result(self):
        return tf.math.divide_no_nan(self.wer_accumulator, self.counter)

    def reset_states(self):
        self.wer_accumulator.assign(0.0)
        self.counter.assign(0.0)


class LERMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the Sentence Error Rate
    """
    def __init__(self, name='LER_metric', **kwargs):
        super(LERMetric, self).__init__(name=name, **kwargs)
        self.LER_accumulator = self.add_weight(name="total_LER", initializer="zeros")
        self.counter = self.add_weight(name="LER_count", initializer="zeros")

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
        distance = tf.edit_distance(decode, y_true_sparse, normalize=False)
        
        correct_words_amount = tf.reduce_sum(tf.cast(tf.not_equal(distance, 0), tf.float32))

        self.LER_accumulator.assign_add(correct_words_amount)
        self.counter.assign_add(tf.cast(len(y_true), 'float32'))

    def result(self):
        return tf.math.divide_no_nan(self.LER_accumulator, self.counter)

    def reset_states(self):
        self.LER_accumulator.assign(0.0)
        self.counter.assign(0.0)