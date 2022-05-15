import tensorflow as tf
from BahdanauAttention import BahdanauAttention
from typing import Any, Tuple
import typing
from shapeChecker import ShapeChecker
import pdb

class RNN(tf.keras.layers.Layer):

	def __init__(self, dim, num_layers=2):
		super(RNN, self).__init__()
		self.dim = dim
		self.num_layers = num_layers
		def layer():
			return tf.keras.layers.GRU(self.dim,
									activation='tanh', 
									recurrent_activation='sigmoid',
									recurrent_dropout = 0,
									unroll = False,
									use_bias = True,
									reset_after = True,
									# Return the sequence and state
									return_sequences=True,
									return_state=True,
									recurrent_initializer='glorot_uniform')
		self._layer_names = ['layer_' + str(i) for i in range(self.num_layers)]
		for name in self._layer_names:
			self.__setattr__(name, layer())

	def call(self, inputs):
		seqs = inputs
		state = None
		for name in self._layer_names:
			rnn = self.__getattribute__(name)
			(seqs, state) = rnn(seqs, initial_state=state)
		return seqs, state


class Decoder(tf.keras.layers.Layer):
	def __init__(self, vocab_size, embedding_dim, dec_units):
		super(Decoder, self).__init__()
		self.dec_units = dec_units
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim

		# For Step 1. The embedding layer convets token IDs to vectors
		self.embedding = tf.keras.layers.Embedding(self.vocab_size,self.embedding_dim)
		self.rnn = RNN(self.dec_units)
		# For Step 2. The RNN keeps track of what's been generated so far.
		#self.gru = tf.keras.layers.GRU(self.dec_units,
		#											activation='tanh', 
		#											recurrent_activation='sigmoid',
		#											recurrent_dropout = 0,
		#											unroll = False,
		#											use_bias = True,
		#											reset_after = True,
		#											# Return the sequence and state
		#											return_sequences=True,
		#											return_state=True,
		#											recurrent_initializer='glorot_uniform')

		#self.bgru = tf.keras.layers.Bidirectional(forward_layer, merge_mode = 'sum')

		# For step 3. The RNN output will be the query for the attention layer.
		self.attention = BahdanauAttention(self.dec_units)

		# For step 4. Eqn. (3): converting `ct` to `at`
		self.Wc = tf.keras.layers.Dense(dec_units, activation=tf.math.tanh,use_bias=False)

		# For step 5. This fully connected layer produces the logits for each
		# output token.
		self.fc = tf.keras.layers.Dense(self.vocab_size)
		#self.attn_weights = tf.zeros(out_enc.shape[1], out_enc.shape[0])



	def call(self, in_char, hidden, encoder_output, prev_attn, target_mask): 
		#shape_checker = ShapeChecker()
		#shape_checker(inputs.new_tokens, ('batch', 't'))
		#shape_checker(inputs.enc_output, ('batch', 's', 'enc_units'))
		#shape_checker(inputs.mask, ('batch', 's'))

		#if state is not None:
		#	shape_checker(state, ('batch', 'dec_units'))
		# Step 1. Lookup the embeddings
		embed_char = self.embedding(in_char)
		#shape_checker(vectors, ('batch', 't', 'embedding_dim'))

		# Step 2. Use the RNN output as the query for the attention over the
		# encoder output.
		context_vector, attn_weights = self.attention(hidden,encoder_output, target_mask)
		#pdb.set_trace()
		#context_vector = tf.squeeze(context_vector, axis = 1)
		
		#shape_checker(context_vector, ('batch', 't', 'dec_units'))
		#shape_checker(attention_weights, ('batch', 't', 's'))

		# Step 4. Eqn. (3): Join the context_vector and rnn_output
		#     [ct; ht] shape: (batch t, value_units + query_units)
		#pdb.set_trace()
		context_and_embed = tf.concat([embed_char, context_vector], axis=1)
		context_and_embed = tf.expand_dims(context_and_embed, axis = 1)
		# Step 2. Process one step with the RNN
		
		rnn_output, latest_state = self.rnn(context_and_embed)

		#shape_checker(rnn_output, ('batch', 't', 'dec_units'))
		#shape_checker(state, ('batch', 'dec_units'))

		# Step 4. Eqn. (3): `at = tanh(Wc@[ct; ht])`
		#attention_vector = self.Wc(context_and_rnn_output)
		#shape_checker(attention_vector, ('batch', 't', 'dec_units'))

		# Step 5. Generate logit predictions:
		logits = self.fc(rnn_output)
		logits = tf.squeeze(logits, axis = 1)
		#shape_checker(logits, ('batch', 't', 'vocab_size'))
		#attn_weights = tf.squeeze(attn_weights, axis = 1)
		return logits, latest_state, attn_weights