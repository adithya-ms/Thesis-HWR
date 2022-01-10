import numpy as np
import os
import tensorflow as tf
#import tensorflow_datasets as tfds
import pathlib
import pdb
import matplotlib.pyplot as plt
import pandas as pd


from xml_processing import get_label_from_filename

def dataloader(input_shape, batch_size):

	traindf=pd.read_csv("./IAM/small_trainLabels.csv", dtype=str)
	testdf=pd.read_csv("./IAM/testLabels.csv", dtype=str)

	train_labels = tf.data.Dataset.from_tensor_slices(dict(traindf))
	test_labels = tf.data.Dataset.from_tensor_slices(dict(testdf))
	datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.,
															validation_split=0.25,
															rotation_range=20,
															height_shift_range=0.2,  
															shear_range=0.2, 
															zoom_range=0.2)
	train_generator=datagen.flow_from_dataframe(traindf,
												directory="../datasets/small_lines/train/",
												x_col="Filename",
												y_col="Label",
												subset="training",
												batch_size=batch_size,
												seed=42,
												shuffle=True,
												class_mode="raw",
												#color_mode = "grayscale",
												target_size=input_shape[:-1])
	
	valid_generator=datagen.flow_from_dataframe(traindf,
												directory="../datasets/small_lines/train/",
												x_col="Filename",
												y_col="Label",
												subset="validation",
												batch_size=batch_size,
												seed=42,
												shuffle=True,
												class_mode="raw",
												#color_mode = "grayscale",
												target_size=input_shape[:-1])
	
	test_datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
	
	test_generator=test_datagen.flow_from_dataframe(testdf,
													directory="./IAM/lines_dataset/test/",
													x_col="Filename",
													y_col=None,
													batch_size=batch_size,
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

if __name__ == '__main__':
	main()