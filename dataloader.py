import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
#import tensorflow_datasets as tfds
import pathlib
import pdb
import matplotlib.pyplot as plt
import pandas as pd

from xml_processing import get_label_from_filename

def dataloader():
	
	traindf=pd.read_csv("./small_trainLabels.csv", dtype=str)
	testdf=pd.read_csv("./small_testLabels.csv", dtype=str)

	datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.,validation_split=0.25)
	train_generator=datagen.flow_from_dataframe(dataframe=traindf,
												directory="./IAM/small_lines/train/",
												x_col="Filename",
												y_col="Label",
												subset="training",
												batch_size=8,
												seed=42,
												shuffle=True,
												class_mode="raw",
												target_size=(50,200))
	
	valid_generator=datagen.flow_from_dataframe(dataframe=traindf,
												directory="./IAM/small_lines/train/",
												x_col="Filename",
												y_col="Label",
												subset="validation",
												batch_size=32,
												seed=42,
												shuffle=True,
												class_mode="raw",
												target_size=(200,1000))
	
	test_datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
	
	test_generator=test_datagen.flow_from_dataframe(dataframe=testdf,
													directory="./IAM/small_lines/test/",
													x_col="Filename",
													y_col=None,
													batch_size=32,
													seed=42,
													shuffle=False,
													class_mode=None,
													target_size=(200,1000))

	#pdb.set_trace()
	plt.figure(figsize=(10, 10))
	img_batch, label_batch = train_generator.next()
	for i in range(3):
		ax = plt.subplot(3, 3, i + 1)
		plt.imshow(img_batch[i])
		plt.title(label_batch[i])
		plt.axis("off")
	#plt.show()

	return train_generator, valid_generator, test_generator

def main():
	train_generator, valid_generator, test_generator = dataloader()


if __name__ == '__main__':
	main()