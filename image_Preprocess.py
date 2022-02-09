import cv2
#import pytesseract
import pdb

def noise_filtering(image):
	#image = cv2.imread('./IAM/complete_monk/train/navis-GilmanLetters-Fam-A_1786-line-007-y1=663-y2=796.png')
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	median = cv2.medianBlur(gray, 5)
	blur = cv2.GaussianBlur(median, (7,7), 0)
	(thresh, thresh_image) = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	return thresh_image

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

def main():
	noise_filtering()

if __name__ == '__main__':
	main()