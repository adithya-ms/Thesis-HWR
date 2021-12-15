import numpy as np
import os
import tensorflow as tf
#import tensorflow_datasets as tfds
import pathlib
import pdb
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices())

from xml_processing import get_label_from_filename

def dataloader(input_shape):
	#pdb.set_trace()
	traindf=pd.read_csv("./trainLabels.csv", dtype=str)
	testdf=pd.read_csv("./testLabels.csv", dtype=str)

	datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.,validation_split=0.25)
	train_generator=datagen.flow_from_dataframe(dataframe=traindf,
												directory="./IAM/lines_dataset/train/",
												x_col="Filename",
												y_col="Label",
												subset="training",
												batch_size=32,
												seed=42,
												shuffle=True,
												class_mode="raw",
												#color_mode = "grayscale",
												target_size=input_shape[:-1])
	
	valid_generator=datagen.flow_from_dataframe(dataframe=traindf,
												directory="./IAM/lines_dataset/train/",
												x_col="Filename",
												y_col="Label",
												subset="validation",
												batch_size=32,
												seed=42,
												shuffle=True,
												class_mode="raw",
												#color_mode = "grayscale",
												target_size=input_shape[:-1])
	
	test_datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
	
	test_generator=test_datagen.flow_from_dataframe(dataframe=testdf,
													directory="./IAM/lines_dataset/test/",
													x_col="Filename",
													y_col=None,
													batch_size=32,
													seed=42,
													shuffle=False,
													class_mode=None,
													#color_mode = "grayscale",
													target_size=input_shape[:-1])

	'''
	plt.figure(figsize=(10, 10))
	img_batch, label_batch = train_generator.next()
	for i in range(3):
		ax = plt.subplot(3, 3, i + 1)
		plt.imshow(img_batch[i])
		plt.title(label_batch[i])
		plt.axis("off")
	#plt.show()
	'''

	return train_generator, valid_generator, test_generator

def main():
	train_generator, valid_generator, test_generator = dataloader((150,600,3))
	count = 0
	pdb.set_trace()

if __name__ == '__main__':
	main()