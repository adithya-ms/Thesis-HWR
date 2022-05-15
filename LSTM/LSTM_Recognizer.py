import tensorflow as tf
from Decoder import Decoder
from Encoder import Encoder
from shapeChecker import ShapeChecker
import pdb
import numpy as np

m = None

def loss_function(real, pred, cer_object, wer_object, tokenizer):
	loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
	mask = tf.math.logical_not(tf.math.equal(real, 0))
	cer_object.reset_states()
	cer_object.update_state(real, pred, tokenizer)
	cer_loss = cer_object.result()

	wer_object.reset_states()
	wer_object.update_state(real, pred, tokenizer)
	wer_loss = wer_object.result()
	
	loss_ = loss_object(tf.cast(real,'int32'), pred)
	mask = tf.cast(mask, dtype=loss_.dtype)
	loss_ *= mask

	return tf.reduce_sum(loss_)/tf.reduce_sum(mask), cer_loss, wer_loss

def accuracy_function(real, pred):
	accuracies = tf.equal(tf.cast(real,'int64'), tf.argmax(pred, axis=2))

	mask = tf.math.logical_not(tf.math.equal(real, 0))
	accuracies = tf.math.logical_and(mask, accuracies)

	accuracies = tf.cast(accuracies, dtype=tf.float32)
	mask = tf.cast(mask, dtype=tf.float32)
	return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

class LSTM_Recognizer(tf.keras.Model):
	def __init__(self, input_shape, embedding_dim, units, vocab_size, batch_size, step_size, step_width, enc_features, use_tf_function=True):
		super().__init__()
		# Build the encoder and decoder
		self.batch_size = batch_size
		self.vocab_size = vocab_size
		self.enc_features = enc_features

		encoder = Encoder(input_shape, embedding_dim, units, batch_size)
		decoder = Decoder(vocab_size,embedding_dim, units)
		self.inp_layer = tf.keras.layers.InputLayer(input_shape=input_shape, batch_size=batch_size)
		self.encoder = encoder
		self.decoder = decoder

		self.use_tf_function = use_tf_function
		
	def _get_masks(self, input_tokens, target_tokens):

		# Convert the text to token IDs
		#self.shape_checker(input_tokens, ('batch', 's'))
		#self.shape_checker(target_tokens, ('batch', 't'))

		# Convert IDs to masks.
		#input_mask = input_tokens != 0
		#self.shape_checker(input_mask, ('batch', 's'))

		target_mask = target_tokens != 0
		#self.shape_checker(target_mask, ('batch', 't'))

		return target_mask


	def call(self, inputs):
		input_features, target_tokens = inputs
		target_mask = (target_tokens != 0)
		#input_mask, target_mask = self._get_masks(input_features, target_tokens)

		max_target_length = tf.shape(target_tokens)[1]
		input_features = self.inp_layer(input_features)

		enc_output, enc_state = self.encoder(input_features)
		
		attns = []
		all_outputs = tf.TensorArray(tf.float32, size=max_target_length)
		attn_weights = tf.zeros([self.batch_size, enc_output.shape[1]])
		
		target_tokens_tr = tf.transpose(target_tokens, perm = [1,0])
		output = target_tokens_tr[0]
		hidden = enc_state

		for index in tf.range(0,max_target_length-1):
			output, hidden, attn_weights = self.decoder(output, hidden, enc_output, attn_weights, target_mask)
			all_outputs = all_outputs.write(index, output)
			output = target_tokens_tr[index+1]
			attns.append(attn_weights) # [(32, 55), ...]

		return all_outputs, attns

	def _loop_step(self, new_tokens, enc_output, dec_state):
		input_token, target_token = new_tokens[:, 0:1], new_tokens[:, 1:2]

		# Run the decoder one step.
		decoder_input = DecoderInput(new_tokens=input_token,
									enc_output=enc_output)

		dec_result, dec_state = self.decoder(decoder_input, state=dec_state)
		#self.shape_checker(dec_result.logits, ('batch', 't1', 'logits'))
		#self.shape_checker(dec_result.attention_weights, ('batch', 't1', 's'))
		#self.shape_checker(dec_state, ('batch', 'dec_units'))

		# `self.loss` returns the total for non-padded tokens
		y_pred = dec_result.logits

		return y_pred