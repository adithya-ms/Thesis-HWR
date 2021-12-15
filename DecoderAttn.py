import tensorflow as tf
from MultiheadAttention import MultiHeadAttention


def point_wise_feed_forward_network(d_model, dff):
	return tf.keras.Sequential([
		tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
		tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
		])

def create_look_ahead_mask(size):
	mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
	return mask  # (seq_len, seq_len)

def create_padding_mask(seq):
	seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

	# add extra dimensions to add the padding
	# to the attention logits.
	return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

class DecoderAttn(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads, dff, rate=0.1):
		super(DecoderAttn, self).__init__()

		self.mha1 = MultiHeadAttention(d_model, num_heads)
		self.mha2 = MultiHeadAttention(d_model, num_heads)

		self.ffn = point_wise_feed_forward_network(d_model, dff)

		self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layernorm4 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

		self.dropout1 = tf.keras.layers.Dropout(rate)
		self.dropout2 = tf.keras.layers.Dropout(rate)
		self.dropout3 = tf.keras.layers.Dropout(rate)
		self.dropout4 = tf.keras.layers.Dropout(rate)

	def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
		# enc_output.shape == (batch_size, input_seq_len, d_model)
		x = self.layernorm1(x)
		attn1, attn_weights_block1 = self.mha1(x, x, x, padding_mask)  # (batch_size, target_seq_len, d_model)
		attn1 = self.dropout1(attn1, training=training)
		out1 = self.layernorm2(attn1 + x)

		out2 = self.ffn(out1)
		out2 = self.dropout2(out2)

		out3 = self.layernorm3(out2 + out1)

		attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out3, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
		attn2 = self.dropout3(attn2, training=training)
		out4 = self.layernorm4(attn2 + out3)  # (batch_size, target_seq_len, d_model)

		ffn_output = self.ffn(out4)  # (batch_size, target_seq_len, d_model)
		ffn_output = self.dropout4(ffn_output, training=training)
		#out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
		#Input ffn_ouputs+attn2 + out3 to softmax here

		return ffn_output, attn_weights_block1, attn_weights_block2
