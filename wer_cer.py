import tensorflow as tf
import tensorflow.keras.backend as K
import pdb
import numpy as np
from datasets import load_metric

class CERMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the Word Piece Error Rate
    """
    def __init__(self, name='CER_metric', **kwargs):
        super(CERMetric, self).__init__(name=name, **kwargs)
        self.cer_accumulator = self.add_weight(name="total_cer", initializer="zeros")
        self.counter = self.add_weight(name="cer_count", initializer="zeros")

    def update_state(self, y_true, y_pred, tokenizer, sample_weight=None):
        input_shape = K.shape(y_pred)
        input_length = tf.ones(shape=input_shape[0]) * K.cast(input_shape[1], 'float32')

        decode, log = K.ctc_decode(y_pred,
                                    input_length,
                                    greedy=True)
        decode[0] = tf.where(decode[0] == -1, np.int64(0) , decode[0])

        decode = K.ctc_label_dense_to_sparse(decode[0], K.cast(input_length, 'int32'))
        y_true_sparse = K.ctc_label_dense_to_sparse(y_true, K.cast(input_length, 'int32'))

        #decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))
        #distance = tf.edit_distance(decode, y_true_sparse, normalize=True)
        distance = self.edit_distance(y_true_sparse, decode, tokenizer)
        self.cer_accumulator.assign_add(tf.reduce_sum(distance))
        self.counter.assign_add(tf.cast(len(y_true), 'float32'))

    def result(self):
        return tf.math.divide_no_nan(self.cer_accumulator, self.counter)
        #return self.cer_accumulator

    def reset_states(self):
        self.cer_accumulator.assign(0.0)
        self.counter.assign(0.0)

    def edit_distance(self, y_true, y_pred, tokenizer):

        y_true_dense = tf.cast(tf.sparse.to_dense(y_true),'int64')
        y_pred_dense = tf.cast(tf.sparse.to_dense(y_pred),'int64')
        insertion = 0
        deletions = 0
        substitutions = 0
        cer_score = []
        for i,label in enumerate(y_true_dense):
            for j, character in enumerate(label):
                if character == 0 and y_pred_dense[i][j] == 0:
                    continue
                elif character == 0 and y_pred_dense[i][j] !=0:
                    deletions += 1
                elif character != 0 and y_pred_dense[i][j] ==0:
                    insertion += 1
                elif character != y_pred_dense[i][j]:
                    substitutions += 1
            
            label_length = np.where(label == 3)
            if label_length[0] > 0:
               label_length = label_length[0]
            else:
                label_length = 100 
            label_cer = float((insertion + deletions + substitutions) / label_length)
            cer_score.append(label_cer)


        #y_pred_strings = tf.strings.unicode_encode(tf.sparse.to_dense(tf.cast(y_pred, 'int32')), 'UTF-8')
        #y_true_strings = tf.strings.unicode_encode(tf.sparse.to_dense(tf.cast(y_true, 'int32')), 'UTF-8')
        #y_pred_strings = tokenizer.en.detokenize(tf.sparse.to_dense(y_pred))
        #y_true_strings = tokenizer.en.detokenize(tf.sparse.to_dense(y_true))

        #cer = load_metric("cer")
        #cer_score = cer.compute(predictions=y_pred_strings, references=y_true_strings)
        return cer_score 




class WERMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the Sentence Error Rate
    """
    def __init__(self, name='WER_metric', **kwargs):
        super(WERMetric, self).__init__(name=name, **kwargs)
        self.wer_accumulator = self.add_weight(name="total_wer", initializer="zeros")
        self.counter = self.add_weight(name="wer_count", initializer="zeros")

    def update_state(self, y_true, y_pred, tokenizer, sample_weight=None):
        input_shape = K.shape(y_pred)
        input_length = tf.ones(shape=input_shape[0]) * K.cast(input_shape[1], 'float32')

        decode, log = K.ctc_decode(y_pred,
                                    input_length,
                                    greedy=True)
        decode[0] = tf.where(decode[0] == -1, np.int64(0), decode[0])
        decode = K.ctc_label_dense_to_sparse(decode[0], K.cast(input_length, 'int32'))
        y_true_sparse = K.ctc_label_dense_to_sparse(y_true, K.cast(input_length, 'int32'))

        #decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))
        
        distance = self.edit_distance(y_true_sparse, decode, tokenizer)
        #pdb.set_trace()
        #distance = tf.edit_distance(decode, y_true_sparse, normalize=True)
        correct_words_amount = tf.reduce_sum(tf.cast(tf.not_equal(distance, 0), tf.float32))

        self.wer_accumulator.assign_add(tf.reduce_sum(distance))
        self.counter.assign_add(tf.cast(len(y_true), 'float32'))

    def result(self):
        return tf.math.divide_no_nan(self.wer_accumulator, self.counter)
        #return self.wer_accumulator

    def reset_states(self):
        self.wer_accumulator.assign(0.0)
        self.counter.assign(0.0)

    def edit_distance(self, y_true, y_pred, tokenizer, normalize=True):
        y_true_dense = tf.cast(tf.sparse.to_dense(y_true),'int64')
        y_pred_dense = tf.cast(tf.sparse.to_dense(y_pred),'int64')
        insertion = 0
        deletions = 0
        substitutions = 0
        wer_score = []

        for i,label in enumerate(y_true_dense):
            space_chars = np.where(label == 32)[0]
            label_length = np.where(label == 3)[0]
            if len(space_chars) > 0:
                pass
            else:
                space_chars = [-1]

            if len(label_length) > 0:
               label_length = label_length[0]
            else:
                label_length = 100

            for j in range(0,len(space_chars)):
                if space_chars[j] == -1:
                    first = 0
                    last = label_length
                elif j == 0:
                    first = 0
                    last = space_chars[j]
                elif j == (len(space_chars) - 1):
                    first = space_chars[j]
                    last = label_length
                else:
                    first = space_chars[j]
                    last = space_chars[j+1]
                #pdb.set_trace()
                if all(label[first:last] == 0) and all(y_pred_dense[i][first:last] == 0):
                    continue
                elif all(label[first:last] == 0) and all(y_pred_dense[i][first:last] != 0):
                    deletions += 1
                elif all(label[first:last] != 0) and all(y_pred_dense[i][first:last] == 0):
                    insertion += 1
                elif any(label[first:last] != y_pred_dense[i][first:last]):
                    substitutions += 1

            word_count = len(space_chars) + 1
            label_wer = float((insertion + deletions + substitutions) / word_count)
            wer_score.append(label_wer)

        #y_pred_strings = tf.strings.unicode_encode(tf.sparse.to_dense(tf.cast(y_pred, 'int32')), 'UTF-8')
        #y_true_strings = tf.strings.unicode_encode(tf.sparse.to_dense(tf.cast(y_true, 'int32')), 'UTF-8')
        #y_pred_strings = tokenizer.en.detokenize(tf.sparse.to_dense(y_pred))
        #y_true_strings = tokenizer.en.detokenize(tf.sparse.to_dense(y_true))
        
        #wer = load_metric("wer")
        #wer_score = wer.compute(predictions=y_pred_strings, references=y_true_strings)
        return wer_score