import tensorflow as tf
from transformer import Transformer, CustomSchedule, loss_function, accuracy_function
from dataloader_monk import dataloader
import pdb
import time
from textPreprocess import preprocess_labels
import os
import datetime
from tensorflow.python.client import device_lib 
from beam_search import beam_search_decoder
from error_metrics import WERMetric, SERMetric

print(device_lib.list_local_devices())
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

num_layers = 4
d_model = 1024
dff = 512
num_heads = 8
dropout_rate = 0.1
input_shape = (64,600,3)
num_epochs = 40
batch_size = 8

wer_object = WERMetric()
ser_object = SERMetric()

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')
test_ser = tf.keras.metrics.Mean(name = 'SER_test')
test_wer = tf.keras.metrics.Mean(name = 'WER_test')

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

checkpoint_path = "D:/Thersis/Results/Final model"
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
	ckpt.restore(ckpt_manager.latest_checkpoint)
	print('Latest checkpoint restored!!')


# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_generator, valid_generator, test_generator = dataloader(input_shape, batch_size)
step_size_test = 1*test_generator.n//test_generator.batch_size
for (batch, (inp, tar)) in enumerate(test_generator):
	tar = preprocess_labels(tar, tokenizer, max_len = 100)
	inp = tf.convert_to_tensor(inp)
	tar = tf.convert_to_tensor(tar)
	tar_inp = tar[:, :-1]
	tar_real = tar[:, 1:]
	predictions, _ = transformer([inp, tar_inp], training = False)
	loss,wer_loss, ser_loss = loss_function(tar_real, predictions, wer_object, ser_object, tokenizer) 
	test_loss(loss)
	test_wer(wer_loss)
	test_ser(ser_loss)
	test_accuracy(accuracy_function(tar_real, predictions))
	if batch > step_size_test:
		break
print(f' Test Loss {test_loss.result():.4f} Test Accuracy {test_accuracy.result():.4f}')
print(f' Test WER {test_wer.result():.4f} Test SER {test_ser.result():.4f}')
