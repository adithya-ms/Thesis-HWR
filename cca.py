import cv2
import pdb
def cca():
	image = cv2.imread('./IAM/complete_monk/train/cliwoc-Adm_177_1189_0001-line-001-y1=11-y2=188.png')
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (7,7), 0)
	thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV +cv2.THRESH_OTSU)[1]
	#pdb.set_trace()
	output = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
	(numLabels, labels, stats, centroids) = output
	for i in range(0, numLabels):
		# if this is the first component then we examine the
		# *background* (typically we would just ignore this
		# component in our loop)
		if i == 0:
			text = "examining component {}/{} (background)".format(
				i + 1, numLabels)
		# otherwise, we are examining an actual connected component
		else:
			text = "examining component {}/{}".format( i + 1, numLabels)
		# print a status message update for the current connected
		# component
		print("[INFO] {}".format(text))
		# extract the connected component statistics and centroid for
		# the current label
		x = stats[i, cv2.CC_STAT_LEFT]
		y = stats[i, cv2.CC_STAT_TOP]
		w = stats[i, cv2.CC_STAT_WIDTH]
		h = stats[i, cv2.CC_STAT_HEIGHT]
		area = stats[i, cv2.CC_STAT_AREA]
		(cX, cY) = centroids[i]
		output = image.copy()
		cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
		cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)
		componentMask = (labels == i).astype("uint8") * 255
		cv2.imshow("Output", output)
		cv2.imshow("Connected Component", componentMask)
		cv2.waitKey(0)


def main():
	cca()
if __name__ == '__main__':
	main()
