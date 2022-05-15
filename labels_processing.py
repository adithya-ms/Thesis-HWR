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
		return os.path.join(folder_path.parent,'trainLabels.csv')
	else:
		file_label_df.to_csv(os.path.join(folder_path.parent,'testLabels.csv'), index=False)
		return os.path.join(folder_path.parent,'testLabels.csv')

def clean_labels(labels_path):
	col_list = ["Filename", "Label"]
	labels_df = pd.read_csv(labels_path, usecols=col_list, encoding = "ISO-8859-1")
	clean_df = pd.DataFrame(columns = ['Filename', 'Label'])
	specials_dict = dict()
		
	for index,batch in enumerate(labels_df['Label']):
		#pdb.set_trace()
		batch = batch.strip()
		if len(batch) <= 5:
			pass
		else:
			labels_df.iloc[index]['Label'] = batch
			clean_df = clean_df.append(labels_df.iloc[[index]], ignore_index=True)

	clean_df.to_csv(labels_path, index=False)


def create_train_test_folders(folder_path, train_folder_path,test_folder_path, train_test_split):
	all_books = os.listdir(folder_path)
	for book in all_books:
		if book[0] == 's':
			continue
		else:
			#pdb.set_trace()
			book_path = os.path.join(folder_path, book)
			all_files = os.listdir(book_path)
			file_counter = int(len(all_files) * train_test_split) 

			test_file_count = 0 
			while test_file_count < file_counter:
				all_files = os.listdir(book_path)
				move_file = random.choice(all_files)
				move_file = move_file[:-3]
				move_img = os.path.join(book_path,move_file + 'png')
				move_txt = os.path.join(book_path,move_file + 'txt')
				dest_img = os.path.join(test_folder_path,move_file + 'png')
				dest_txt = os.path.join(test_folder_path,move_file + 'txt')

				pathlib.Path(move_img).rename(dest_img)
				test_file_count += 1
				pathlib.Path(move_txt).rename(dest_txt)
				test_file_count += 1

			train_files = os.listdir(book_path)
			for move_file in train_files:
				move_file = move_file[:-3]
				move_img = os.path.join(book_path,move_file + 'png')
				move_txt = os.path.join(book_path,move_file + 'txt')
				dest_img = os.path.join(train_folder_path,move_file + 'png')
				dest_txt = os.path.join(train_folder_path,move_file + 'txt')

				pathlib.Path(move_img).rename(dest_img)
				pathlib.Path(move_txt).rename(dest_txt)


def main():
	#labels_path = 'IAM/complete_monk/all_trainLabels.csv'
	#clean_labels(labels_path)
	#all_books = "D:\\Thersis\\datasets\\Monk-Transcribed-Lines-Selection-2021-v1.0"
	train_folder_path = pathlib.Path('./IAM/aug_monk/train')
	test_folder_path = os.path.join(train_folder_path.parent, 'test')
	#create_train_test_folders(all_books, train_folder_path,test_folder_path,train_test_split = 0.2)
	train_labels_path = create_labels_csv(train_folder_path, is_train = True)
	test_labels_path =create_labels_csv(test_folder_path, is_train = False)
	clean_labels(train_labels_path)

if __name__ == "__main__":
	main()