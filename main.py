import tensorflow as tf
from transformer import Transformer, CustomSchedule, loss_function, accuracy_function
from dataloader import dataloader
import pdb
import time
from textPreprocess import preprocess_labels
import os
import datetime
from tensorflow.python.client import device_lib 
from beam_search import beam_search_decoder


print(device_lib.list_local_devices())
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
tf.debugging.experimental.enable_dump_debug_info('logs/gradient_tape', tensor_debug_mode="NO_TENSOR", circular_buffer_size=-1)

train_step_signature = [
	tf.TensorSpec(shape=(None, None, None,None), dtype=tf.float32),
	tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

#@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
	tar_inp = tar[:, :-1]
	tar_real = tar[:, 1:]

	with tf.GradientTape() as tape:
		predictions, _ = transformer([inp, tar_inp], training = True)
		#predictions = beam_search_decoder(predictions, top_k = 1)
		#predictions = tf.nn.ctc_beam_search_decoder(tf.transpose(predictions, perm = [2,0,1]),
		# 											[predictions.shape[1]] *predictions.shape[0],
		# 											beam_width=100, top_paths=1)
		loss = loss_function(tar_real, predictions)
		
		#wer = 
		#cer = 

	gradients = tape.gradient(loss, transformer.trainable_variables)
	optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

	train_loss(loss)
	train_accuracy(accuracy_function(tar_real, predictions))

	return gradients

num_layers = 4
d_model = 1024
dff = 512
num_heads = 8
dropout_rate = 0.1
input_shape = (32,200,3)
num_epochs = 20
batch_size = 16

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
train_wer = tf.keras.metrics.Mean(name = 'WER_metric')
train_cer = tf.keras.metrics.Mean(name = 'CER_metric')
tokenizer = tf.saved_model.load('bert-en-subword-tokenizer')
vocab_size = tokenizer.en.get_vocab_size().numpy()
transformer = Transformer(
	num_layers=num_layers,
	d_model=d_model,
	num_heads=num_heads,
	dff=dff,
	vocab_size=vocab_size,
	#target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
	pe_input=1000,
	pe_target=1000,
	input_shape = input_shape,
	rate=dropout_rate)

checkpoint_path = "./checkpoints/train"
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
#if ckpt_manager.latest_checkpoint:
#	ckpt.restore(ckpt_manager.latest_checkpoint)
#	print('Latest checkpoint restored!!')

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
graph_log_dir = 'logs/gradient_tape/' + current_time + '/graph'

graph_summary_writer = tf.summary.create_file_writer(graph_log_dir)
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.


train_generator, valid_generator, test_generator = dataloader(input_shape, batch_size)
for epoch in range(num_epochs):
	start = time.time()

	train_loss.reset_states()
	train_accuracy.reset_states()

	step_size_train = 2*train_generator.n//train_generator.batch_size
	for (batch, (inp, tar)) in enumerate(train_generator):
		tar = preprocess_labels(tar, tokenizer, max_len = 100)
		tf.summary.trace_on(graph=True, profiler=True)

		gradients = train_step(inp, tar)
		with graph_summary_writer.as_default():
			tf.summary.trace_export(name="my_func_trace", step=0, profiler_outdir=graph_log_dir)

		if batch % 5 == 0:
			print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

		if batch > step_size_train:
			break
	with train_summary_writer.as_default():
		tf.summary.scalar('loss', train_loss.result(), step=epoch)
		tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
		l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
		for gradient, variable in zip(gradients, transformer.trainable_variables):
			tf.summary.histogram("gradients/" + variable.name, l2_norm(gradient), step = 0)
			tf.summary.histogram("variables/" + variable.name, l2_norm(variable), step = 0)

	if (epoch + 1) % 5 == 0:
		ckpt_save_path = ckpt_manager.save()
		print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

	print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

	print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
	