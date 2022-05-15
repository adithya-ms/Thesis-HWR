import tensorflow as tf
import matplotlib.pyplot as plt
import pdb
from transformer import Transformer, CustomSchedule, loss_function
import sys
import matplotlib.gridspec as gridspec
sys.path.append("..")
from textPreprocess import preprocess_labels
from dataloader_monk import dataloader

from error_metrics import WERMetric, SERMetric


class HWRecognizer(tf.Module):
	def __init__(self, tokenizer, transformer):
		self.tokenizer = tokenizer
		self.transformer = transformer

	def __call__(self, input_img, tar_real, max_length=20):
		input_img = tf.convert_to_tensor(input_img)
		
		# as the target is english, the first token to the transformer should be the
		# english start token.
		start_end = self.tokenizer.en.tokenize(['']).values
		start = start_end[0][tf.newaxis]
		end = start_end[1][tf.newaxis]

		# `tf.TensorArray` is required here (instead of a python list) so that the
		# dynamic-loop can be traced by `tf.function`.
		output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
		#pdb.set_trace()
		output_array = output_array.write(0, start)
		
		wer_object = WERMetric()
		ser_object = SERMetric()
		#pdb.set_trace()
		for i in tf.range(max_length):
			output = tf.transpose(output_array.stack())
			predictions, _ = self.transformer([input_img, output], training=False)
			#loss,wer_loss, ser_loss = loss_function(tar_real, predictions, wer_object, ser_object)
				
			# select the last token from the seq_len dimension
			predictions = predictions[-1, :, :]  

			predicted_id = tf.argmax(predictions, axis=-1)

			# concatentate the predicted_id to the output which is given to the decoder
			# as its input.
			#for i in range(0,predicted_id.shape[1]):
			output_array = output_array.write(i+1, [predicted_id[-1]])

			if [predicted_id[-1]] == end:
				break

		output = tf.transpose(output_array.stack())
		# output.shape (1, tokens)
		text = self.tokenizer.en.detokenize(output)[0]  # shape: ()

		tokens = self.tokenizer.en.lookup(output)[0]

		# `tf.function` prevents us from using the attention_weights that were
		# calculated on the last iteration of the loop. So recalculate them outside
		# the loop.
		_, attention_weights = self.transformer([input_img, output[:,:-1]], training=False)

		return text, tokens, attention_weights
		

def print_hw_text(input_img, tokens, ground_truth):
	imgplot = plt.imshow(input_img)
	plt.title(str("Prediction: ") + str(tokens.numpy().decode("utf-8")) + str("\nGround truth: ") + str(ground_truth))
	plt.show()

def plot_attention_head(translated_tokens, attention, tar_text, input_img, tokens):
	# The plot is of the attention when a token was generated.
	# The model didn't generate `<START>` in the output. Skip it.
	pdb.set_trace()
	fig = plt.figure(figsize=(30, 12))
	outer = gridspec.GridSpec(3, 1, wspace=0.2, hspace=0.2)

	ax1 = plt.Subplot(fig, outer[0])
	im = ax1.imshow(input_img)
	ax1.set_title(str("Prediction: ") + str(tokens.numpy().decode("utf-8")) + str("\nGround truth: ") + str(tar_text))
	#global fig
	fig.add_subplot(ax1)
	
	inner = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=outer[1], wspace=0.1, hspace=0.1)

	for j in range(8):
		ax2 = plt.Subplot(fig, inner[j])
		ax2.matshow(attention[j])
		ax2.set_yticks(range(len(translated_tokens)))
		labels = [label.decode('utf-8') for label in translated_tokens.numpy()]
		ax2.set_yticklabels(labels)
		fig.add_subplot(ax2)

	fig.show()

	'''
	fig, (ax1, ax2) = plt.subplots(2, 1)
	
	
	 = plt.subplots(4,2)

	for attention in attention_heads:
		ax2.matshow(attention)
		#ax.set_xticks(range(len(in_tokens)))
		ax2.set_yticks(range(len(translated_tokens)))
		#ax.set_xticks(ground_truth)


		#labels = [label.decode('utf-8') for label in in_tokens.numpy()]
		#ax.set_xticklabels(labels, rotation=90)

		labels = [label.decode('utf-8') for label in translated_tokens.numpy()]
		ax2.set_yticklabels(labels)
	fig.show()
	'''

def main():
	num_layers = 2
	d_model = 256
	dff = 128
	num_heads = 8
	dropout_rate = 0.1
	input_shape = (64,800,3)
	batch_size = 1
	dataset = 'IAM'
	datapath = '../IAM/lines_dataset'

	tokenizer = tf.saved_model.load('../bert-en-subword-tokenizer-'+dataset)
	vocab_size = tokenizer.en.get_vocab_size().numpy()

	transformer = Transformer(num_layers=num_layers,
								d_model=d_model,
								num_heads=num_heads,
								dff=dff,
								vocab_size=vocab_size,
								pe_input=1000,
								pe_target=1000,
								input_shape = input_shape,
								rate=dropout_rate)

	checkpoint_path = "D:/Thersis/Results/Final model/IAM"
	learning_rate = CustomSchedule(d_model)
	optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
	ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

	ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

	# if a checkpoint exists, restore the latest checkpoint.
	if ckpt_manager.latest_checkpoint:
		ckpt.restore(ckpt_manager.latest_checkpoint)
		print('Latest checkpoint restored!!')

	recognizer = HWRecognizer(tokenizer, transformer)
	
	train_generator, valid_generator, test_generator = dataloader(datapath,input_shape, batch_size)
	#step_size_test = 1*test_generator.n//test_generator.batch_size
	for (batch, (input_img, tar_text)) in enumerate(test_generator):
		tar = preprocess_labels(tar_text, tokenizer, max_len = 100)
		tar = tf.convert_to_tensor(tar)
		tar_real = tar[:, 1:]
		translated_text, translated_tokens, attention_weights = recognizer(input_img, tar_real)
		input_img = tf.reshape(input_img,input_shape)
		#print_hw_text(input_img, translated_text, tar_text)

		translated_tokens = translated_tokens[:-1]
		head = 0
		# shape: (batch=1, num_heads, seq_len_q, seq_len_k)
		attention_heads = tf.squeeze(
		attention_weights['decoder_layer4_block2'], 0)

		plot_attention_head(translated_tokens, attention_heads, tar_text, input_img, translated_text)


	
		
	
if __name__ == '__main__':
	main()