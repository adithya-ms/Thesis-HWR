from dataloader import dataloader
import pandas as pd
import pdb
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def get_tokenizer():
	pdb.set_trace()
	train_labels=pd.read_csv("./small_trainLabels.csv", dtype=str)
	tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
	tk.fit_on_texts(train_labels['Label'])
	char_dict = get_alphabet(train_labels['Label'])
	
	# Use char_dict to replace the tk.word_index
	tk.word_index = char_dict.copy()
	
	# Add 'UNK' to the vocabulary
	tk.start_token = '<S>'
	tk.end_token = '<E>'
	tk.pad_token = '<P>'

	tk.word_index[tk.oov_token] = max(char_dict.values()) + 1
	#Add start and end tokens to the vocabulary:
	tk.word_index[tk.start_token] = max(char_dict.values()) + 2
	tk.word_index[tk.end_token] = max(char_dict.values()) + 3	

	processed_labels = preprocess_labels(train_labels['Label'], tk)
	return tk

def get_alphabet(train_labels):
	alphabet = []
	for label in train_labels:
		for character in label:
			if character in alphabet:
				pass
			else:
				alphabet.append(character)

	char_dict = {}
	for i, char in enumerate(alphabet):
	    char_dict[char] = i + 1

	return char_dict


def preprocess_labels(label_batch, tk, max_len = 100):
	# Convert string to index
	padded_labels = [] 
	for label in label_batch:
		label = tk.start_token + label + tk.end_token
		#while len(label) < max_len:
		#	label = label + tk.pad_token
		padded_labels.append(label[:100])
	pdb.set_trace()
	padded_labels = tk.texts_to_sequences(padded_labels)

	# Padding
	padded_labels = pad_sequences(padded_labels, maxlen=100, padding='post')
	
	# Convert to numpy array
	padded_labels = np.array(padded_labels, dtype='float32')
	
	return padded_labels


def main():
	tokenizer = get_tokenizer()

if __name__ == '__main__':
	main()