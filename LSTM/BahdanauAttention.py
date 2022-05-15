import tensorflow as tf
from shapeChecker import ShapeChecker
import pdb

class BahdanauAttention(tf.keras.layers.Layer):
	def __init__(self, units):
		super().__init__()
		# For Eqn. (4), the  Bahdanau attention
		self.W1 = tf.keras.layers.Dense(units, use_bias=False)
		self.W2 = tf.keras.layers.Dense(units, use_bias=False)
		self.out = tf.keras.layers.Dense(1)
		self.attention = tf.keras.layers.Attention(use_scale = True)

	def call(self, query, value, target_mask):
		#shape_checker = ShapeChecker()
		#shape_checker(query, ('batch', 't', 'query_units'))
		#shape_checker(value, ('batch', 's', 'value_units'))
		#shape_checker(mask, ('batch', 's'))
		#pdb.set_trace()
		# From Eqn. (4), `W1@ht`.
		query = tf.expand_dims(query,axis = 1)

		w1_query = self.W1(query)

		#shape_checker(w1_query, ('batch', 't', 'attn_units'))

		# From Eqn. (4), `W2@hs`.
		w2_key = self.W2(value)
		#shape_checker(w2_key, ('batch', 's', 'attn_units'))

		query_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)
		value_mask = target_mask
		res_attn = tf.math.tanh(w2_key + w1_query)
		out_attn = self.out(res_attn)
		out_attn = tf.squeeze(out_attn,2)
		
		#context_vector = self.attention(inputs = [w1_query, w2_key, value],)
															#mask=[query_mask, target_mask],
															#return_attention_scores = True)

		attention_weights = tf.nn.softmax(out_attn)
		attention_weights = tf.matmul(tf.transpose(value, [0,2,1]), attention_weights, transpose_b=True)
		
		#shape_checker(context_vector, ('batch', 't', 'value_units'))
		#shape_checker(attention_weights, ('batch', 't', 's'))

		return out_attn, attention_weights