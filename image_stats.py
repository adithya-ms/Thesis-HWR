from PIL import Image
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt

def main():
	imag_dir = './IAM/complete_monk/train/'
	img_files = os.listdir(imag_dir)
	image_sizes = np.empty([len(img_files),2])
	batch = 0
	for file in img_files:
		file_name = os.path.join('./IAM/complete_monk/train/',file)
		with Image.open(file_name) as im:
			for i in [0,1]:
				image_sizes[batch][i] = im.size[i]
		batch += 1

	plt.hist(image_sizes[:,0], bins=100)
	plt.gca().set(title='Lengths Histogram', ylabel='Frequency');
	plt.show()

	plt.hist(image_sizes[:,1], bins=50)
	plt.gca().set(title='Width Histogram', ylabel='Frequency');
	plt.show()

	
if __name__ == '__main__':
	main()