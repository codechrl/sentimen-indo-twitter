#!/usr/bin/env python
# coding: utf-8

# In[9]:


import re
import json
import gensim
import pickle
import Sastrawi
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.tokenize import word_tokenize
from keras.models import model_from_json
from nltk.tokenize import WordPunctTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# In[2]:


def tweet_cleaner(text):
    tok = WordPunctTokenizer()
    x=text
    # hapus rt
    cl = re.sub(r'\s*RT\s*@[^:]*:.*', '', x)
    cl = re.sub(r'\s*rt\s*@[^:]*:.*', '', cl)
    # hapus mention
    cl = re.sub(r'@[A-Za-z0-9]([^:\s]+)+', '', cl)
    # hapus link
    cl = re.sub(r'https?://[A-Za-z0-9./]+', '', cl)
    # hapus hashtag
    cl = re.sub(r'(?:\s|^)#[A-Za-z0-9\-\.\_]+(?:\s|$)', '', cl)
    # kata ulang
    cl = re.sub(r'\w*\d\w*', '', cl)
    cl = re.sub(r'\b(\w+)(\1\b)+', r'\1', cl)
    # hapus simbol
    cl = re.sub(r'[^a-zA-Z]', ' ', cl)
    # lower
    cl=cl.lower()
    # format teks 
    cl=tok.tokenize(cl)
    cl=(" ".join(cl))
    return cl

def stopword(text):
    # stopwords sastrawi
    factory = StopWordRemoverFactory()

    # tambah stopwords ke dict sastrawi
    more_stopwords=[line.strip() for line in open('lib/more_stopwords.txt')]
    factory.get_stop_words()+more_stopwords
    stopwords = factory.create_stop_word_remover()

    # hapus stopwords
    result = stopwords.remove(text)
    return result

def stem(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    result= stemmer.stem(text)
    return result

# In[4]:


def tokenn(input_clean):
    with open('lib/tokenizer_indo.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)                                 
    sequences = tokenizer.texts_to_sequences(input_clean)               
    len(tokenizer.word_index)
    #
    length = []
    for x in input_clean:
        length.append(len(x.split()))
    max(length)
    return sequences

# In[6]:


def pad(sequences):
    x_train_seq = pad_sequences(sequences, maxlen=70)
    return x_train_seq


# In[7]:
# load model
json_file = open('lib/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('lib/weights.hdf5')

global graph
graph = tf.get_default_graph()

def predict(text,clean):
    input_clean=text
    print(input_clean)
    
    if clean!=True:
        # text cleaning
        input_clean = tweet_cleaner(input_clean)
        print(input_clean)
        # stopwords
        input_clean = stopword(input_clean)
        print(input_clean)
        # stemming
        input_clean = stem(input_clean)
        print(input_clean)
    # simpan ke dataframe
    df=pd.DataFrame([input_clean], columns=['text'])
    input_clean=df.text
    # tokenizing
    sequences=tokenn(input_clean)
    # padding
    input_ready=pad(sequences)
    # predict classes
    with graph.as_default():
        prediction = loaded_model.predict_classes(input_ready).tolist()
    
    return json.dumps(prediction)


# In[10]:


#input= "@AmericanAir so bad can't take this anymore. never again"
#print(predict(input))


# In[ ]:


