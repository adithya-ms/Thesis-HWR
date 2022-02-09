import pandas as pd
import os
import pdb
import pathlib
import random
'''
def get_label_from_file(text_filepath):
	all_files = os.listdir(text_filepath)
	all_txt_files = []
	for file in all_files:
		if file[-4:] == '.txt':
			 all_txt_files.append(file)

	for file
'''
def create_labels_csv(folder_path, is_train = True):
	file_label_df = pd.DataFrame(columns = ['Filename', 'Label'])
	folder_path = pathlib.Path(folder_path)
	#labels_dir = pathlib.Path("./IAM/labels/")
	for label_file in folder_path.iterdir():
		if label_file.suffix == '.png':
			continue
		with open(label_file) as f:
			label_string = f.read()

		file_label_df = file_label_df.append({'Filename':label_file.name[:-4], 'Label': label_string}, ignore_index=True)

	if is_train is True:
		file_label_df.to_csv(os.path.join(folder_path.parent,'trainLabels.csv'), index=False)
	else:
		file_label_df.to_csv(os.path.join(folder_path.parent,'testLabels.csv'), index=False)

def create_train_test_folders(folder_path, train_test_split):
	all_files = os.listdir(folder_path)
	file_counter = len(all_files) * train_test_split
	test_folder_path = os.path.join(folder_path.parent, 'test') 
	if not os.path.exists(test_folder_path):
		os.mkdir(test_folder_path)
	test_file_count = len(os.listdir(test_folder_path))
	while test_file_count != file_counter:
		all_files = os.listdir(folder_path)
		move_file = random.choice(all_files)
		move_file = move_file[:-3]
		move_img = os.path.join(folder_path,move_file + 'png')
		move_txt = os.path.join(folder_path,move_file + 'txt')
		dest_img = os.path.join(test_folder_path,move_file + 'png')
		dest_txt = os.path.join(test_folder_path,move_file + 'txt')
		pathlib.Path(move_img).rename(dest_img)
		pathlib.Path(move_txt).rename(dest_txt)
		test_file_count = len(os.listdir(test_folder_path))

	return test_folder_path


def main():
	train_folder_path = pathlib.Path('./IAM/complete_monk/train')
	test_folder_path = os.path.join(train_folder_path.parent, 'test')
	#test_folder_path = create_train_test_folders(train_folder_path,train_test_split = 0.2)
	create_labels_csv(train_folder_path, is_train = True)
	create_labels_csv(test_folder_path, is_train = False)

if __name__ == "__main__":
	main()