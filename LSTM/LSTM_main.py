import tensorflow as tf
import pdb
import time
import os
import datetime
from tensorflow.python.client import device_lib
from LSTM_Recognizer import LSTM_Recognizer, loss_function, accuracy_function
import sys
sys.path.append("..")
from textPreprocess import preprocess_labels, vectorize_text
from image_Preprocess import noise_filtering
from dataloader_monk import dataloader 
from error_metrics import WERMetric, SERMetric
from tensorflow.keras.applications.vgg19 import VGG19
import matplotlib.pyplot as plt

print(device_lib.list_local_devices())
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

#tf.debugging.experimental.enable_dump_debug_info('logs/gradient_tape', tensor_debug_mode="NO_TENSOR", circular_buffer_size=-1)

embedding_dim = 50
units = 50
dropout_rate = 0.1
input_shape = (50,800,3)
batch_size = 4
num_epochs = 10
step_size = 20
step_width = 100
few_shot = 0.1
dataset = 'Monk'
datapath = "../IAM/complete_monk"


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
	def __init__(self, d_model, warmup_steps=2000):
		super(CustomSchedule, self).__init__()

		self.d_model = d_model
		self.d_model = tf.cast(self.d_model, tf.float32)

		self.warmup_steps = warmup_steps

	def __call__(self, step):
		arg1 = tf.math.rsqrt(step)
		arg2 = step * (self.warmup_steps ** -1.5)

		return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

initial_learning_rate = 0.0002
#CustomSchedule(1024)
#plt.plot(initial_learning_rate(tf.range(40000, dtype=tf.float32)))
#plt.ylabel("Learning Rate")
#plt.xlabel("Train Step")
#plt.show()

print('Embedding_dim = ',embedding_dim)
print('units = ',units)
print('input_shape = ',input_shape)
print('batch_size = ',batch_size)
print('num_epochs = ',num_epochs)
print('few_shot = ',few_shot)
print('dataset = ',dataset)
print('initial_learning_rate = ', initial_learning_rate)

WER_object = WERMetric()
SER_object = SERMetric()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
train_SER = tf.keras.metrics.Mean(name = 'SER_Train')
train_WER = tf.keras.metrics.Mean(name = 'WER_Train')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.Mean(name='val_accuracy')
val_SER = tf.keras.metrics.Mean(name = 'SER_Val')
val_WER = tf.keras.metrics.Mean(name = 'WER_Val')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.Mean(name='test_accuracy')
test_SER = tf.keras.metrics.Mean(name = 'SER_test')
test_WER = tf.keras.metrics.Mean(name = 'WER_test')

tokenizer = tf.saved_model.load('../bert-en-subword-tokenizer-'+dataset)
vocab_size = tokenizer.en.get_vocab_size().numpy()
encoder_output = VGG19(include_top = False, weights = 'imagenet', input_shape = input_shape).get_layer('block5_conv4').output.shape
enc_features = encoder_output[1]*encoder_output[3]

recognizer = LSTM_Recognizer(
	input_shape,
	embedding_dim, 
	units,
	vocab_size,
	batch_size,
	step_size,
	step_width,
	enc_features,
	use_tf_function=True)

checkpoint_path = "./checkpoints/train_LSTM/"
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=200, decay_rate=1.1, staircase=True
)

optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
ckpt = tf.train.Checkpoint(recognizer=recognizer, optimizer=optimizer)
recognizer.compile(optimizer=optimizer, loss=loss_function)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.

#if ckpt_manager.latest_checkpoint:
#	ckpt.restore(ckpt_manager.latest_checkpoint)
#	print('Latest checkpoint restored!!')

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'lstm_logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'lstm_logs/gradient_tape/' + current_time + '/test'
graph_log_dir = 'lstm_logs/gradient_tape/' + current_time + '/graph'

graph_summary_writer = tf.summary.create_file_writer(graph_log_dir)
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
val_summary_writer = tf.summary.create_file_writer(test_log_dir)


# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
	tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
	tf.TensorSpec(shape=(None, None), dtype=tf.int64),
	#tf.TensorSpec(shape=(None, None), dtype=tf.float32),
]
count = 0
#@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
	#tf.keras.backend.clear_session()
	tar_inp = tar[:, :-1]
	tar_real = tar[:, 1:]

	with tf.GradientTape() as tape:
		predictions, _ = recognizer([inp, tar_inp] )
		#pdb.set_trace()
		predictions = predictions.stack()
		predictions = tf.transpose(predictions, [1,0,2])
		#pdb.set_trace()
		#Check y_pred before computing this loss
		loss,WER_loss, SER_loss = loss_function(tar_real, predictions, WER_object, SER_object, tokenizer)

	gradients = tape.gradient(loss, recognizer.trainable_variables)
	optimizer.apply_gradients(zip(gradients, recognizer.trainable_variables))
	#global count
	#count +=1 
	#print(count+1)
	train_loss(loss)
	train_accuracy(accuracy_function(tar_real, predictions))
	train_WER(WER_loss)
	train_SER(SER_loss)

	return gradients


train_generator, valid_generator, test_generator = dataloader(datapath,input_shape, batch_size)
for epoch in range(num_epochs):
	start = time.time()

	train_loss.reset_states()
	train_accuracy.reset_states()

	train_WER.reset_states()
	train_SER.reset_states()

	val_loss.reset_states()
	val_accuracy.reset_states()

	val_WER.reset_states()
	val_SER.reset_states()

	step_size_train = few_shot*train_generator.n//train_generator.batch_size
	for (batch, (inp, tar)) in enumerate(train_generator):
		tar = preprocess_labels(tar, tokenizer, max_len = 100)
		#tar = vectorize_text(tar)
		inp = noise_filtering(inp)
		#pdb.set_trace()

		inp = tf.convert_to_tensor(inp)
		tar = tf.convert_to_tensor(tar)

		#tar_inp = tar[:, :-1]
		tar_real = tar[:, 1:]
		
		if epoch == 0 and batch == 0:
			tf.summary.trace_on(graph=True, profiler=False)

		gradients = train_step(inp, tar)
		if epoch == 0 and batch == 0:
			print(recognizer.summary())
			with graph_summary_writer.as_default():
				tf.summary.trace_export(name="train_trace", step=1, profiler_outdir=graph_log_dir)
				tf.summary.trace_off()
		if batch % 10 == 0:
			print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}\
				WER {train_WER.result():.4f} SER {train_SER.result():.4f}')
		if batch > step_size_train:
			break
	#pdb.set_trace()
	#print("LearningRate: ", lr_schedule)


	step_size_val = few_shot*valid_generator.n//valid_generator.batch_size
	for (batch, (inp, tar)) in enumerate(valid_generator):
		tar = preprocess_labels(tar, tokenizer, max_len = 100)
		inp = noise_filtering(inp)
		#inp = preprocess_images(inp, step_width, step_size)
		inp = tf.convert_to_tensor(inp)
		tar = tf.convert_to_tensor(tar)

		tar_inp = tar[:, :-1]
		tar_real = tar[:, 1:]
		#pdb.set_trace()
		predictions = recognizer([inp, tar_inp] )
		predictions = predictions[0].stack()
		predictions = tf.transpose(predictions, [1,0,2])
		loss,WER_loss, SER_loss = loss_function(tar_real, predictions, WER_object, SER_object, tokenizer)
		
		val_loss(loss)
		val_WER(WER_loss)
		val_SER(SER_loss)
		val_accuracy(accuracy_function(tar_real, predictions))
		
		if batch > step_size_val:
			break

	with train_summary_writer.as_default():
		tf.summary.scalar('loss', train_loss.result(), step=epoch)
		tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
		tf.summary.scalar('WER', train_WER.result(), step=epoch)
		tf.summary.scalar('SER', train_SER.result(), step=epoch)
		l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
		for gradient, variable in zip(gradients, recognizer.trainable_variables):
			tf.summary.histogram("gradients/" + variable.name, l2_norm(gradient), step = 0)
			tf.summary.histogram("variables/" + variable.name, l2_norm(variable), step = 0)

	with val_summary_writer.as_default():
		tf.summary.scalar('loss', val_loss.result(), step=epoch)
		tf.summary.scalar('accuracy', val_accuracy.result(), step=epoch)
		tf.summary.scalar('Val_SER', val_SER.result(), step=epoch)
		tf.summary.scalar('Val_WER', val_WER.result(), step=epoch)

	if (epoch + 1) % 1 == 0:
		ckpt_save_path = ckpt_manager.save()
		print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

	print(f'Epoch {epoch + 1} Train Loss {train_loss.result():.4f} Train Accuracy {train_accuracy.result():.4f}')
	print(f'Epoch {epoch + 1} Train WER {train_WER.result():.4f} Train SER {train_SER.result():.4f}')
	print(f'Epoch {epoch + 1} Validation Loss {val_loss.result():.4f} Validation Accuracy {val_accuracy.result():.4f}')
	print(f'Epoch {epoch + 1} Validation WER {val_WER.result():.4f} Validation SER {val_SER.result():.4f}')
	print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')


step_size_test = few_shot*test_generator.n//test_generator.batch_size
for (batch, (inp, tar)) in enumerate(test_generator):
	tar = preprocess_labels(tar, tokenizer, max_len = 100)
	#inp = preprocess_images(inp, step_width, step_size)
	inp = noise_filtering(inp)

	inp = tf.convert_to_tensor(inp)
	tar = tf.convert_to_tensor(tar)

	#tar_inp = tar[:, :-1]
	tar_real = tar[:, 1:]

	predictions = recognizer([inp, tar_inp])
	predictions = predictions[0].stack()
	predictions = tf.transpose(predictions, [1,0,2])
	loss,WER_loss, SER_loss = loss_function(tar_real, predictions, WER_object, SER_object, tokenizer) 
	
	test_loss(loss)
	test_WER(WER_loss)
	test_SER(SER_loss)
	test_accuracy(accuracy_function(tar_real, predictions))
	
	if batch > step_size_test:
		break
print(f' Test Loss {test_loss.result():.4f} Test Accuracy {test_accuracy.result():.4f}')
print(f' Test WER {test_WER.result():.4f} Test SER {test_SER.result():.4f}')
