import pandas as pd
import pdb
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
import numpy as np
import os
import tensorflow_text as text
import pathlib
import re

class CustomTokenizer(tf.Module):
	def __init__(self, reserved_tokens, vocab_path):
		self.tokenizer = text.BertTokenizer(vocab_path, lower_case=False)
		self._reserved_tokens = reserved_tokens
		self._vocab_path = tf.saved_model.Asset(vocab_path)

		vocab = pathlib.Path(vocab_path).read_text().splitlines()
		self.vocab = tf.Variable(vocab)

		## Create the signatures for export:   

		# Include a tokenize signature for a batch of strings. 
		self.tokenize.get_concrete_function(
			tf.TensorSpec(shape=[None], dtype=tf.string))

		# Include `detokenize` and `lookup` signatures for:
		#   * `Tensors` with shapes [tokens] and [batch, tokens]
		#   * `RaggedTensors` with shape [batch, tokens]
		self.detokenize.get_concrete_function(
			tf.TensorSpec(shape=[None, None], dtype=tf.int64))
		self.detokenize.get_concrete_function(
			tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

		self.lookup.get_concrete_function(
			tf.TensorSpec(shape=[None, None], dtype=tf.int64))
		self.lookup.get_concrete_function(
			tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

		# These `get_*` methods take no arguments
		self.get_vocab_size.get_concrete_function()
		self.get_vocab_path.get_concrete_function()
		self.get_reserved_tokens.get_concrete_function()

	@tf.function
	def tokenize(self, strings):
		enc = self.tokenizer.tokenize(strings)
		# Merge the `word` and `word-piece` axes.
		enc = enc.merge_dims(-2,-1)
		enc = add_start_end(enc)
		return enc

	@tf.function
	def detokenize(self, tokenized):
		words = self.tokenizer.detokenize(tokenized)
		return cleanup_text(self._reserved_tokens, words)

	@tf.function
	def lookup(self, token_ids):
		return tf.gather(self.vocab, token_ids)

	@tf.function
	def get_vocab_size(self):
		return tf.shape(self.vocab)[0]

	@tf.function
	def get_vocab_path(self):
		return self._vocab_path

	@tf.function
	def get_reserved_tokens(self):
		return tf.constant(self._reserved_tokens)

reserved_tokens=["[PAD]", "[UNK]", "[START]", "[END]"]
bert_tokenizer_params=dict(lower_case=False)

def build_tokenizer(labels_path):

	if not os.path.exists("vocab.txt"):
		create_alphabet_file(labels_path)

	tokenizers = tf.Module()
	tokenizers.en = CustomTokenizer(reserved_tokens, 'vocab.txt')

	tf.saved_model.save(tokenizers, 'bert-en-subword-tokenizer')	

def create_alphabet_file(labels_path):

	train_labels_path = os.path.join(labels_path, 'trainLabels.csv')
	train_labels = pd.read_csv(train_labels_path)
	train_labels = train_labels["Label"]
	train_labels = tf.data.Dataset.from_tensor_slices(train_labels)

	bert_vocab_args = dict(vocab_size = 8000,
							reserved_tokens=reserved_tokens,
							bert_tokenizer_params=bert_tokenizer_params,
							learn_params={},)
	vocab = bert_vocab.bert_vocab_from_dataset(train_labels.batch(1000).prefetch(2),**bert_vocab_args)
	write_vocab_file('vocab.txt', vocab)

def write_vocab_file(filepath, vocab):
	with open(filepath, 'w') as f:
		for token in vocab:
			print(token, file=f)

def cleanup_text(reserved_tokens, token_txt):
	# Drop the reserved tokens, except for "[UNK]".
	bad_tokens = [re.escape(tok) for tok in reserved_tokens if tok != "[UNK]"]
	bad_token_re = "|".join(bad_tokens)

	bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
	result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

	# Join them into strings.
	result = tf.strings.reduce_join(result, separator=' ', axis=-1)
	return result

def add_start_end(ragged):
	START = tf.argmax(tf.constant(reserved_tokens) == "[START]")
	END = tf.argmax(tf.constant(reserved_tokens) == "[END]")
	count = ragged.bounding_shape()[0]
	starts = tf.fill([count,1], START)
	ends = tf.fill([count,1], END)
	return tf.concat([starts, ragged, ends], axis=1)

def preprocess_labels(label_batch, tk, max_len = 100):
	# Convert string to index
	label_batch = tk.en.tokenize(label_batch).to_tensor()
	return label_batch

def main():
	labels_path = "./IAM/complete_monk/"
	tokenizer = build_tokenizer(labels_path)

if __name__ == '__main__':
	main()