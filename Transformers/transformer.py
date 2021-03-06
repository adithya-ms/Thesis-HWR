import tensorflow as tf
from DecoderAttn import create_padding_mask, create_look_ahead_mask
from VFEncoder import VFEncoder
from text_transcriber import Transcriber
import pdb
import numpy as np
import enchant

def loss_function(real, pred, cer_object, wer_object, tokenizer):
	loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

	mask = tf.math.logical_not(tf.math.equal(real, 0))
	cer_object.reset_states()
	cer_object.update_state(real, pred, tokenizer)
	cer_loss = cer_object.result()

	wer_object.reset_states()
	wer_object.update_state(real, pred, tokenizer)
	wer_loss = wer_object.result()
	
	loss_ = loss_object(real, pred)
	mask = tf.cast(mask, dtype=loss_.dtype)
	loss_ *= mask

	return tf.reduce_sum(loss_)/tf.reduce_sum(mask), cer_loss, wer_loss

def accuracy_function(real, pred):
	accuracies = tf.equal(real, tf.argmax(pred, axis=2))

	mask = tf.math.logical_not(tf.math.equal(real, 0))
	accuracies = tf.math.logical_and(mask, accuracies)

	accuracies = tf.cast(accuracies, dtype=tf.float32)
	mask = tf.cast(mask, dtype=tf.float32)
	return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


class Transformer(tf.keras.Model):
	def __init__(self, num_layers, d_model, num_heads, dff, pe_input, pe_target, input_shape, vocab_size, rate=0.1):
		super().__init__()
		self.encoder = VFEncoder(num_layers, d_model, num_heads, dff,pe_input, input_shape, rate)

		self.decoder = Transcriber(num_layers, d_model, num_heads, dff, pe_target, vocab_size, rate)

		self.final_layer = tf.keras.layers.Dense(vocab_size, activation='softmax')

	def call(self, inputs, training):
		# Keras models prefer if you pass all your inputs in the first argument
		inp, tar = inputs
		#pdb.set_trace()
		enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)

		enc_output = self.encoder(inp, training)  # (batch_size, inp_seq_len, d_model)

		# dec_output.shape == (batch_size, tar_seq_len, d_model)

		dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

		final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

		return final_output, attention_weights

	def create_masks(self, inp, tar):
		# Encoder padding mask
		#pdb.set_trace()
		enc_padding_mask = create_padding_mask(inp)

		# Used in the 2nd attention block in the decoder.
		# This padding mask is used to mask the encoder outputs.
		dec_padding_mask = create_padding_mask(inp)

		# Used in the 1st attention block in the decoder.
		# It is used to pad and mask future tokens in the input received by
		# the decoder.
		look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
		dec_target_padding_mask = create_padding_mask(tar)
		look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

		return enc_padding_mask, look_ahead_mask, dec_padding_mask

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
	def __init__(self, d_model, warmup_steps=4000):
		super(CustomSchedule, self).__init__()

		self.d_model = d_model
		self.d_model = tf.cast(self.d_model, tf.float32)

		self.warmup_steps = warmup_steps

	def __call__(self, step):
		arg1 = tf.math.rsqrt(step)
		arg2 = step * (self.warmup_steps ** -1.5)

		return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
