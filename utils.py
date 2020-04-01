import os
import numpy as np 
import pandas as pd
from tqdm.notebook import tqdm
import string
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, Dense, Dropout, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

#Path of the directory
path = os.getcwd()

#READ THE TOKENS FROM THE FILE 
def load_description(path, filename):
    file = open(path + "\\" + filename)
    data = file.read()
    desc = dict()
    for line in data.split("\n"):
        tokens = line.split()
        if len(line) < 2:
            continue
        image_id, image_desc = tokens[0], tokens[1:]
        image_id = image_id.split(".")[0]
        image_desc = " ".join(image_desc)
        if image_id not in desc.keys():
            desc[image_id] = list()
        desc[image_id].append(image_desc)
    return desc

#CLEAINING THE DATA
def clean_description(dict_input):
    table = str.maketrans("","", string.punctuation)
    print("Processing Started")
    for key, value in tqdm(dict_input.items(), total = len(dict_input)):
        for i in range(len(value)):
            #tokens
            desc_tokens = value[i].split()
            #table translation
            desc_tokens = [i.translate(table) for i in desc_tokens]
            #lower translate
            desc_tokens = [i.lower() for i in desc_tokens]
            #remove simple a and s
            desc_tokens = [i for i in desc_tokens if len(i)>1]
            #remove numerical tokens
            desc_tokens = [i for i in desc_tokens if i.isalpha()]
            value[i] = " ".join(desc_tokens)
    print("Processing Done !!!!!!")
    return dict_input

#up till now unique_token are 8763
# Remove tokens where frequency less than 10
def preprocessed_unique_words(dict_input):
    all_train_captions = []
    for key, value in dict_input.items():
        for cap in value:
            all_train_captions.append(cap)
    word_count_threshold = 10
    word_counts = {}
    for sent in all_train_captions:
        for w in sent.split(" "):
            word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print("Preprocessed vocab_size : ", len(vocab))
    return vocab

#Save description
def save_description(input_dict, filename):
    lines = list()
    for key, value in input_dict.items():
        for desc in value:
            lines.append(key + " " + desc)
    data = '\n'.join(lines)
    file = open(filename, "w")
    file.write(data)
    file.close()


#Retrieve train_image file_name and more
def retieve_image_name(path, filename):
    file = open(path + "\\" + filename)
    data = file.read()
    train_list = []
    for line in data.split("\n"):
        image_name = line.split(".")[0]
        train_list.append(image_name)
    return train_list

#ADD startseq AND endseq TO THE CAPTIONS
def create_mapping_train_captions(images_name, captions_data):
    train_image_capt = dict()
    for key, value in captions_data.items():
        buffer = list()
        if key in images_name:
            for i in range(len(value)):
                buffer.append("startseq " + value[i] + " endseq")
            train_image_capt[key] = buffer
    return train_image_capt

def load_photo_features(filename, dataset):
    all_features = list()
    with ((open(filename, 'rb'))) as file:
        all_features.append(pickle.load(file))
    all_features = all_features[0]
    features = {k:all_features[k] for k in dataset}
    return features




# COMPUTATIONALLY HEAVY FUNCTION USE ONCE AND PICKLE SAVE IT
# GET AUTOMATIC FEATURE REPRESENTATIONS FROM INSEPTIONv3
def load_and_get_afe(path, images_name):
    start = time.time()
    files_name = os.listdir(path)
    model =  tf.keras.applications.InceptionV3(weights = 'imagenet')
    model_final = tf.keras.models.Model(model.input, model.layers[-2].output)
    model_final.trainable = False
    encoded_images_dict = {}
    for i in tqdm(files_name):
        check_name = i.split(".")[0]
        if check_name in images_name:
            total_path = path + "\\" + i
            #image = tf.io.read_file(image)
            #image = tf.io.decode_image(image, channels=3)
            img = tf.keras.preprocessing.image.load_img(total_path, target_size = (299, 299))
            image = tf.keras.preprocessing.image.img_to_array(img)
            image = np.expand_dims(image, axis = 0)
            image = tf.keras.applications.inception_v3.preprocess_input(image)
            image_out_model = model_final.predict(image, verbose = 0)
            image_out_model_reshaped = np.reshape(image_out_model, image_out_model.shape[1])
            assert image_out_model_reshaped.shape == (2048, )
            encoded_images_dict[check_name] = image_out_model_reshaped
    end = time.time()
    print("Total time taken : {0}".format(end - start))
    return encoded_images_dict

def load_and_get_afe_1(path):
    start = time.time()
    files_name = os.listdir(path)
    model =  tf.keras.applications.InceptionV3(weights = 'imagenet')
    model_final = tf.keras.models.Model(model.input, model.layers[-2].output)
    model_final.trainable = False
    encoded_images_dict = {}
    for i in tqdm(files_name):
        total_path = path + "\\" + i
        check_name = i.split(".")[0]
        #image = tf.io.read_file(image)
        #image = tf.io.decode_image(image, channels=3)
        img = tf.keras.preprocessing.image.load_img(total_path, target_size = (299, 299))
        image = tf.keras.preprocessing.image.img_to_array(img)
        image = np.expand_dims(image, axis = 0)
        image = tf.keras.applications.inception_v3.preprocess_input(image)
        image_out_model = model_final.predict(image, verbose = 0)
        #image_out_model_reshaped = np.reshape(image_out_model, image_out_model.shape[1])
        #assert image_out_model_reshaped.shape == (2048, )
        #encoded_images_dict[check_name] = image_out_model_reshaped
        encoded_images_dict[check_name] = image_out_model
    end = time.time()
    print("Total time taken : {0}".format(end - start))
    return encoded_images_dict


#create tokenzier
def create_tokenizer(input_dict):
    lines = to_lines(input_dict)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# CREATING THE DICTIONARIES idxtoword and wordtoidx
def word_id(vocab):
    idxtoword = {}
    wordtoidx = {}
    idx = 1
    for w in vocab:
        wordtoidx[w] = idx
        idxtoword[idx] = w
        idx += 1
    return idxtoword, wordtoidx

def to_lines(input_dict):
    all_desc = list()
    for keys in input_dict.keys():
        [all_desc.append(d) for d in input_dict[keys]]
    return all_desc

def max_length(input_dict):
    lines = to_lines(input_dict)
    list_word = [len(line.split()) for line in lines]
    return max(list_word)


def create_sequences(tokenizer, max_length, descriptions, photos, vocab_size):
	X1, X2, y = list(), list(), list()
	# walk through each image identifier
	for key, desc_list in tqdm(descriptions.items()):
		# walk through each description for the image
		for desc in desc_list:
			# encode the sequence
			seq = tokenizer.texts_to_sequences([desc])[0]
			# split one sequence into multiple X,y pairs
			for i in range(1, len(seq)):
				# split into input and output pair
				in_seq, out_seq = seq[:i], seq[i]
				# pad input sequence
				in_seq = tf.keras.preprocessing.sequence.pad_sequences([in_seq], maxlen=max_length)[0]
				# encode output sequence
				out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0]
				# store
				X1.append(photos[key][0])
				X2.append(in_seq)
				y.append(out_seq)
	return np.array(X1), np.array(X2), np.array(y)

# LOADING GLOVE FILE FOR EMBEDDING MATRIX
def load_glove(path):
    file = open(path, encoding='utf-8')
    embedding_dict = dict()
    for line in file:
        values = line.split()
        words = values[0]
        coeff = np.asarray(values[1:], dtype = np.float32)
        embedding_dict[words] = coeff
    file.close()
    return embedding_dict

# CREATING THE EMBEDDING MATRIX
def create_embedding_matrix(vocab_size, path_to_glove, tokenizer_wordtoidx):
    glove_dict = load_glove(path_to_glove)
    embedding_dims = 200
    embedding_matrix = np.zeros((vocab_size, embedding_dims))
    for word, i in wordtoidx.items():
        embedding_vector = glove_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

# THE MODEL TO BE USED
def model_nn(input_shape_image, vocab_size, max_length, embed_dims):
    #FOR THE IMAGE INPUT 
    inputs1 = Input(input_shape_image)
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    ##RNN PART
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, embed_dims, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    ##DECODER
    decoder1 = tf.keras.layers.Add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    #MODEL
    model = Model([inputs1, inputs2], outputs)
    print(model.summary())
    return model