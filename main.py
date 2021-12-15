import tensorflow as tf
from transformer import Transformer, CustomSchedule, loss_function, accuracy_function
from dataloader import dataloader
import pdb
import time
from textPreprocess import get_tokenizer, preprocess_labels
from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices())

train_step_signature = [
	tf.TensorSpec(shape=(None, None, None,None), dtype=tf.float32),
	tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
	tar_inp = tar[:, :-1]
	tar_real = tar[:, 1:]

	with tf.GradientTape() as tape:
		predictions, _ = transformer([inp, tar_inp], training = True)
		loss = loss_function(tar_real, predictions)

	gradients = tape.gradient(loss, transformer.trainable_variables)
	optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

	train_loss(loss)
	train_accuracy(accuracy_function(tar_real, predictions))

num_layers = 4
d_model = 1024
dff = 512
num_heads = 8
dropout_rate = 0.1
input_shape = (100,400,3)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

transformer = Transformer(
	num_layers=num_layers,
	d_model=d_model,
	num_heads=num_heads,
	dff=dff,
	#input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
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

EPOCHS = 20

# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_generator, valid_generator, test_generator = dataloader(input_shape)
for epoch in range(EPOCHS):
	start = time.time()

	train_loss.reset_states()
	train_accuracy.reset_states()
	tokenizer = get_tokenizer()
	# inp -> portuguese, tar -> english
	step_size_train = train_generator.n//train_generator.batch_size
	for (batch, (inp, tar)) in enumerate(train_generator):
		tar = preprocess_labels(tar, tokenizer, max_len = 100)
		train_step(inp, tar)

		if batch % 5 == 0:
			print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

		if batch > step_size_train:
			break
	if (epoch + 1) % 5 == 0:
		ckpt_save_path = ckpt_manager.save()
		print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

	print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

	print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
	