import numpy as np
import os
import tensorflow as tf
#import tensorflow_datasets as tfds
import pathlib
import pdb
import matplotlib.pyplot as plt
import pandas as pd


from xml_processing import get_label_from_filename

def dataloader(data_path,input_shape, batch_size):

	traindf=pd.read_csv(os.path.join(data_path,"trainLabels.csv"), dtype=str)
	testdf=pd.read_csv(os.path.join(data_path,"testLabels.csv"), dtype=str)

	train_labels = tf.data.Dataset.from_tensor_slices(dict(traindf))
	test_labels = tf.data.Dataset.from_tensor_slices(dict(testdf))
	datagen=tf.keras.preprocessing.image.ImageDataGenerator(#rescale=1./255.,
															validation_split=0.25,
															rotation_range=2,
															height_shift_range=0.2,  
															shear_range=0.2, 
															zoom_range=0.2)
	train_generator=datagen.flow_from_dataframe(traindf,
												directory=os.path.join(data_path,"train/"),
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
												directory=os.path.join(data_path,"train/"),
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
													directory=os.path.join(data_path,"test/"),
													x_col="Filename",
													y_col="Label",
													batch_size=batch_size,
													seed=42,
													shuffle=True,
													class_mode="raw",
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
	train_generator, valid_generator, test_generator = dataloader((50,300,3),8)
	count = 0

if __name__ == '__main__':
	main()