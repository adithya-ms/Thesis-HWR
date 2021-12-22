import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from Attention_block import EncoderAttn
from positional_embeddings import positional_encoding
import pdb
from DecoderAttn import create_padding_mask

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()



class VFEncoder(tf.keras.layers.Layer):
	def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, input_shape, rate=0.1):
		super(VFEncoder, self).__init__()

		self.d_model = d_model
		self.num_layers = num_layers
		self.inp_shape = input_shape
        
		self.resnet50 = ResNet50(input_shape = self.inp_shape, weights='imagenet', pooling=max, include_top = False)
		self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

		self.attn_layers = [EncoderAttn(d_model, num_heads, dff, rate) for _ in range(num_layers)]

		self.dense1 = tf.keras.layers.Dense(d_model)
		self.dense2 = tf.keras.layers.Dense(d_model)
		self.dense3 = tf.keras.layers.Dense(d_model)

		self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

		self.dropout = tf.keras.layers.Dropout(rate)

	def call(self, x, training):

		# adding resnet features embedding and position encoding.
		x = self.resnet50(x)  # (batch_size, input_seq_len, d_model)

		length, width, feature = x.shape[1::]
		x = x.reshape(-1,width,length*feature)

		x = self.dense1(x)

		x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

		seq_len = x.shape[1]
		x += self.pos_encoding[:, :seq_len, :]
        
		x = self.dense2(x)
		x = self.layernorm(x)

		mask = create_padding_mask(x)
		for i in range(self.num_layers):
			x = self.attn_layers[i](x, training, mask)
        
		x = self.dense3(x)
		x = self.dropout(x, training=training)

		return x  # (batch_size, input_seq_len, d_model)
