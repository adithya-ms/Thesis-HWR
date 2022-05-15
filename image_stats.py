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
	count = 0
	aspect_ratio = dict()
	aspect_ratio_list = []
	for file in img_files:
		file_name = os.path.join('./IAM/complete_monk/train/',file)
		with Image.open(file_name) as im:
			for i in [0,1]:
				image_sizes[batch][i] = im.size[i]
		aspect_ratio[file] = round(image_sizes[batch][0]/ image_sizes[batch][1],2)
		aspect_ratio_list.append(round(image_sizes[batch][0]/ image_sizes[batch][1],2))
		batch += 1

	#pdb.set_trace()
	for key, value in aspect_ratio.items():
		if value < 3 or value > 27:
			count += 1
			print(key," : ", value)

	pdb.set_trace()
	print("Images with aspect Ratio outside bounds: ", count)
	print("Mean of Aspect Ratios:",np.mean(aspect_ratio_list))
	print("Standard Deviation of Aspect Ratios:",np.std(aspect_ratio_list))
	plt.hist(aspect_ratio.values(), bins=100, color = 'g')
	plt.gca().set(title='Aspect Ratio Histogram', ylabel='Frequency', xlabel='Aspect Ratio Value')
	plt.show()

	print("Mean of Width:",np.mean(image_sizes[:,0]))
	print("Standard Deviation of Width:",np.std(image_sizes[:,0]))
	plt.hist(image_sizes[:,0], bins=100, color = 'g')
	plt.gca().set(title='Width Histogram', ylabel='Frequency', xlabel='Number of Pixels')
	plt.show()

	print("Mean of Height:",np.mean(image_sizes[:,1]))
	print("Standard Deviation of Height:",np.std(image_sizes[:,1]))
	plt.hist(image_sizes[:,1], bins=100, color = 'g')
	plt.gca().set(title='Height Histogram', ylabel='Frequency', xlabel='Number of Pixels')
	plt.show()



	
if __name__ == '__main__':
	main()