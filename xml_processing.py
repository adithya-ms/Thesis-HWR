import xml.etree.ElementTree as ET
import pathlib
import pdb
import pandas as pd

def get_label_from_filename(img_filename):
	xml_dir = pathlib.Path("./IAM/xml/")
	#Remove .png extension from image
	img_filename = img_filename[:-4]

	img_slice = int(img_filename[-2:])
	img_filename = img_filename[:-3] + '.xml'
	xml_path = xml_dir.joinpath(img_filename)
	if xml_path.exists():
		tree = ET.parse(xml_path)
		root = tree.getroot()
		if root[1].tag == 'handwritten-part':
			if root[1][img_slice].tag == 'line':
				return root[1][img_slice].attrib['text']
			else:
				raise Exception("Line part not found within XML")
		else:
			raise Exception("Handwritten part not found within XML")
	else:
		raise Exception("XML File not found")	

def create_labels_csv(is_train = True):
	file_label_df = pd.DataFrame(columns = ['Filename', 'Label'])
	if is_train is True:
		img_dir = pathlib.Path("./IAM/lines_dataset/train/")
	else:
	 	img_dir = pathlib.Path("./IAM/lines_dataset/test/")
	
	xml_dir = pathlib.Path("./IAM/xml/")
	for img_file in img_dir.iterdir():
		label_string = get_label_from_filename(img_file.name)
		file_label_df = file_label_df.append({'Filename':img_file.name, 'Label': label_string}, ignore_index=True)

	if is_train is True:
		file_label_df.to_csv('trainLabels.csv', index=False)
	else:
		file_label_df.to_csv('testLabels.csv', index=False)


def main():
	create_labels_csv(is_train = True)
	create_labels_csv(is_train = False)

if __name__ == "__main__":
	main()