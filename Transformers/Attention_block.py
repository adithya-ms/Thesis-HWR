import tensorflow as tf
from MultiheadAttention import MultiHeadAttention
import pdb



def point_wise_feed_forward_network(d_model, dff):
	return tf.keras.Sequential([
		tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
		tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
	])

class EncoderAttn(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads, dff, rate=0.1):
		super(EncoderAttn, self).__init__()

		self.heads = num_heads

		self.mha = MultiHeadAttention(d_model, num_heads)
		self.ffn = point_wise_feed_forward_network(d_model, dff)

		self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

		self.dropout1 = tf.keras.layers.Dropout(rate)
		self.dropout2 = tf.keras.layers.Dropout(rate)

	def call(self, x, training, mask):

		attn_output, _ = self.mha(x, x, x, None, look_ahead = False)  # (batch_size, input_seq_len, d_model)
		attn_output = self.dropout1(attn_output, training=training)
		out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

		ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
		ffn_output = self.dropout2(ffn_output, training=training)
		out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

		return out2

def main():
	multihead_self_attn(features)

if __name__ == '__main__':
	main()