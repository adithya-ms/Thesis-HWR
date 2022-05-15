import cv2
import numpy as np
import pdb

def noise_filtering(image_batch):

	return_images = []
	for image in image_batch:
		pdb.set_trace()

		cv2.imwrite('Inputimage.jpg', image)
		bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		cv2.imwrite('BWimage.jpg', bw_image)
		median = cv2.medianBlur(bw_image, 3)
		#blur = cv2.GaussianBlur(median, (3,3), 0)
		cv2.imwrite('Medianimage.jpg', median)
		#cv2.imwrite('Gaussianimage.jpg', blur)
		(thresh, thresh_image) = cv2.threshold(median, 200, 255, cv2.THRESH_BINARY_INV)
		cv2.imwrite('Thresholdedimage.jpg', thresh_image)
		#cv2.imwrite('RGBimage.jpg', backtorgb)
		norm_image = cv2.normalize(thresh_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		backtorgb = cv2.cvtColor(norm_image,cv2.COLOR_GRAY2RGB)
		#cv2.imwrite('Normalizedimage.jpg', norm_image)
		return_images.append(backtorgb)
	return return_images


def bounding_box():
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,13))
	dilate = cv2.dilate(thresh, kernel, iterations = 1)
	#cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])
	for c in cnts:
		x, y, width, height = cv2.boundingRect(c)
		cv2.rectangle(image, (x,y),(x+width,y+height), (36, 255,12), 2)
	cv2.imshow('Show', image)
	cv2.waitKey(0)
	cv2.imwrite('test_gray.png', image)

def preprocess_images(input_img):
	inp_patches = []
	num_patches = np.uint32((input_img.shape[2] - step_width) / step_size)
	for index,image in enumerate(input_img):
		patches = []
		count = 0
		for i in range(0,image.shape[1], step_size):
			if i+step_width > image.shape[1]:
				break
			else:
				patch = image[:,i:i+step_width,:]
				patch = noise_filtering(patch)
				patches.append(patch)
				count += 1
		inp_patches.append(patches)
	inp_patches = np.array(inp_patches)

	return inp_patches




def main():
	noise_filtering()

if __name__ == '__main__':
	main()