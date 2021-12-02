import os
import pathlib
import shutil

def main():
	dest_dir = pathlib.Path("./IAM/lines_dataset/")
	data_dir = pathlib.Path("./IAM/lines/")
	dir_list = [x for x in data_dir.iterdir() if x.is_dir()]
	for subdir in dir_list:
		subdir_list = [x for x in subdir.iterdir() if x.is_dir()]
		for sub_subdir in subdir_list:
			files_list = [x for x in sub_subdir.iterdir() if x.is_file()]
			for file in files_list:
				to_file = dest_dir.joinpath(file.name)
				shutil.copy(file, to_file)
				

if __name__ == '__main__':
	main()