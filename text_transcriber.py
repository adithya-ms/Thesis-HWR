import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from DecoderAttn import DecoderAttn
from positional_embeddings import positional_encoding
import pdb


class Transcriber(tf.keras.layers.Layer):
	def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, vocab_size, rate=0.1):
		super(Transcriber, self).__init__()

		self.d_model = d_model
		self.num_layers = num_layers

		self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
		self.dense = tf.keras.layers.Dense(d_model)
		self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

		self.dec_layers = [DecoderAttn(d_model, num_heads, dff, rate) for _ in range(num_layers)]
		self.dropout = tf.keras.layers.Dropout(rate)

	def call(self, x, enc_output, training, look_ahead_mask, padding_mask):

		seq_len = tf.shape(x)[1]
		attention_weights = {}
		x = self.embedding(x)
		#x = self.dense(x)  # (batch_size, target_seq_len, d_model)
		x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
		x += self.pos_encoding[:, :seq_len, :]


		for i in range(self.num_layers):
			x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

			attention_weights[f'decoder_layer{i+1}_block1'] = block1
			attention_weights[f'decoder_layer{i+1}_block2'] = block2

		# x.shape == (batch_size, target_seq_len, d_model)
		return x, attention_weights

