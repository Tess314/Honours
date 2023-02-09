#!/usr/bin/env python
# coding: utf-8

# # Library Imports

# In[58]:


import collections
import pandas as pd
import numpy as np
from keras.models import Model
from collections import Counter
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import GRU, Input, TimeDistributed, Dense, Activation


# # Read in Data

# In[2]:


english_file = open("C:\\Users\\tessw\\OneDrive\\Documents\\University\\Honours\\english.txt", "r")
hausa_file = open("C:\\Users\\tessw\\OneDrive\\Documents\\University\\Honours\\hausa.txt", "r")

english_sentences = english_file.read()
hausa_sentences = hausa_file.read()

print('Dataset Loaded')


# # Analysis

# In[3]:


#Sample sentences

stop_char = '.'

english_line = english_sentences.split(stop_char, 1)
hausa_line = hausa_sentences.split(stop_char, 1)

print('English line : ' + english_line[0])
print()
print('Hausa line : ' + hausa_line[0])


# In[4]:


#Vocabulary

split_eng = english_sentences.split()
Counters_found_eng = Counter(split_eng)
most_occur_eng = Counters_found_eng.most_common(5)
print("Most common English words in dataset: ")
print(most_occur_eng)

split_hau = hausa_sentences.split()
Counters_found_hau = Counter(split_hau)
most_occur_hau = Counters_found_hau.most_common(5)
print("Most common Hausa words in dataset: ")
print(most_occur_hau)


# # Text Representation

# In[22]:


#Tokenising
def tokenize(x):
    x_tk = Tokenizer(char_level = False)
    x_tk.fit_on_texts(x)
    return x_tk.texts_to_sequences(x), x_tk

text_tokenized_eng, text_tokenizer_eng = tokenize(split_eng)
text_tokenized_hau, text_tokenizer_hau = tokenize(split_hau)
#print(text_tokenizer_eng.word_index)
print()
#print(text_tokenizer_hau.word_index)


# In[24]:


#Padding
def pad(x, length=None):
    if length is None:
        length = max([len(sentence) for sentence in x])
    return pad_sequences(x, maxlen = length, padding = 'post')

english_padding = pad(text_tokenized_eng)
hausa_padding = pad(text_tokenized_hau)

#for sample_i, (token_sent, pad_sent) in enumerate(zip(text_tokenized_eng, test_pad)):
    #print('Sequence {} in x'.format(sample_i + 1))
    #print('  Input:  {}'.format(np.array(token_sent)))
    #print('  Output: {}'.format(pad_sent))


# # Pre-processing

# In[30]:


def preprocess(x, y):
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)
    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)
    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)
    return preprocess_x, preprocess_y, x_tk, y_tk

preproc_english_sentences, preproc_hausa_sentences, english_tokenizer, hausa_tokenizer =\
preprocess(english_sentences, hausa_sentences)
    
max_english_sequence_length = preproc_english_sentences.shape[1]
max_hausa_sequence_length = preproc_hausa_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
hausa_vocab_size = len(hausa_tokenizer.word_index)
print('Data Preprocessed')
print("Max English sentence length:", max_english_sequence_length)
print("Max Hausa sentence length:", max_hausa_sequence_length)
print("English vocabulary size:", english_vocab_size)
print("Hausa vocabulary size:", hausa_vocab_size)


# # Machine Translation Model

# In[32]:


#Ids back to text

def logits_to_text(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'
    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

print('`logits_to_text` function loaded.')


# In[59]:


#RNN model function

def simple_model(input_shape, output_sequence_length, english_vocab_size, hausa_vocab_size):
    learning_rate = 1e-3
    input_seq = Input(input_shape[1:])
    rnn = GRU(64, return_sequences = True)(input_seq)
    logits = TimeDistributed(Dense(hausa_vocab_size))(rnn)
    model = Model(input_seq, Activation('softmax')(logits))
    model.compile(loss = sparse_categorical_crossentropy, 
                 optimizer = Adam(learning_rate), 
                 metrics = ['accuracy'])
    return model


# In[60]:


#Using the RNN

tmp_x = pad(preproc_english_sentences, max_hausa_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_hausa_sentences.shape[-2], 1))
# Train the neural network
simple_rnn_model = simple_model(tmp_x.shape, max_hausa_sequence_length, english_vocab_size, hausa_vocab_size)
simple_rnn_model.fit(tmp_x, preproc_hausa_sentences, batch_size=1024, epochs=10, validation_split=0.2)
# Print prediction(s)
print(logits_to_text(simple_rnn_model.predict(tmp_x[:1])[0], hausa_tokenizer))


# # Evaluation Metrics

# In[ ]:




