import tensorflow as tf
from ConvReader import ConvReader
from shapeChecker import ShapeChecker
import pdb
from tensorflow.keras.layers import TimeDistributed
from resnet import ResNet50
#from tensorflow.keras.applications.resnet50 import ResNet50
import sys
sys.path.append("..")
from positional_embeddings import positional_encoding
#from tensorflow.keras.applications.vgg19 import VGG19

class RNN(tf.keras.layers.Layer):

	def __init__(self, dim, num_layers=1):
		super(RNN, self).__init__()
		self.dim = dim
		self.num_layers = num_layers
		def layer():
			forward_layer = tf.keras.layers.GRU(self.dim,
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
			return tf.keras.layers.Bidirectional(forward_layer, merge_mode = 'sum')

		self._layer_names = ['layer_' + str(i) for i in range(self.num_layers)]
		for name in self._layer_names:
			self.__setattr__(name, layer())

	def call(self, inputs):
		seqs = inputs
		state = None
		for name in self._layer_names:
			rnn = self.__getattribute__(name)
			seqs, state1, state2 = rnn(seqs, initial_state=state)
		return seqs, state1 + state2

class Encoder(tf.keras.layers.Layer):
	def __init__(self, input_shape, d_model, enc_units, batch_size):
		super(Encoder, self).__init__()
		self.enc_units = enc_units
		self.inp_shape = input_shape
		self.d_model = d_model

		# The embedding layer converts tokens to vectors
		
		self.embedding = ResNet50(input_shape = self.inp_shape, weights='imagenet', pooling=max, include_top = False)
		#self.resnet50.trainable = False
		
		self.pos_encoding = positional_encoding(100, 512)
		self.dense = tf.keras.layers.Dense(512)
		self.rnn = RNN(self.enc_units)	


	def call(self, x, state=None):
		
		#shape_checker = ShapeChecker()
		#shape_checker(tokens, ('batch', 's'))

		# 2. The embedding layer looks up the embedding for each token.
		pdb.set_trace()
		vectors = self.embedding(x)
		
		vectors = tf.reshape(vectors,[-1, vectors.shape[2], vectors.shape[1]*vectors.shape[3]]) 
		#shape_checker(vectors, ('batch', 's', 'embed_dim'))

		# 3. The GRU processes the embedding sequence.
		#    output shape: (batch, s, enc_units)
		#    state shape: (batch, enc_units)
		vectors_dense = self.dense(vectors)
		seq_len = vectors_dense.shape[1]
		vectors_dense += self.pos_encoding[:, :seq_len, :]

		output, state = self.rnn(vectors_dense) #, initial_state=state)
		
		#shape_checker(output, ('batch', 's', 'enc_units'))
		#shape_checker(state, ('batch', 'enc_units'))

		# 4. Returns the new sequence and its state.
		return output, state