import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.layers import TimeDistributed
import pdb

class ConvReader(tf.keras.layers.Layer):
	def __init__(self, d_model, input_shape, batch_size, step_size, step_width):
		super(ConvReader, self).__init__()
		self.d_model = d_model
		self.inp_shape = input_shape
		self.step_size = step_size
		self.step_width = step_width

		windows = int((self.inp_shape[1] - step_width) / step_size + 1)
		
		self.inputs = (windows,input_shape[0], step_width, input_shape[2])
		self.model = tf.keras.Sequential()
		self.model.add(tf.keras.Input(shape=self.inputs))
		self.model.add(TimeDistributed(layers.Conv2D(20, (5, 5), activation='relu')))
		self.model.add(TimeDistributed(layers.Conv2D(50, (5, 5), activation='relu')))
		self.model.add(TimeDistributed(layers.MaxPooling2D((2,2))))
		self.model.add(TimeDistributed(layers.Flatten()))
		self.model.add(TimeDistributed(layers.Dense(d_model)))

	def call(self, x):
		x = self.model(x)
		#x = self.maxpool(x)
		#x = self.conv2d_2(x)
		#x = self.maxpool(x)
		#x = self.flatten(x)
		#x = self.dense(x)

		return x
