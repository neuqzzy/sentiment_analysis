# author - mpk001
#exit Mar 26 2017
import numpy as np
import pandas as pd
import re

from bs4 import BeautifulSoup

import sys
import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Masking
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Sequential, Model

SENTENCE_NUM = 25000
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)  #删除多余的换行符，使用re.sub
    return string.strip().lower()  #将大写字母返回为小写字母

data_train = pd.read_csv('labeledTrainData.tsv', sep='\t')  #读取训练集，以tab作为分隔符
print(data_train.shape)  #打印训练集的长度

texts = []
labels = [] #创建两个空列表

for idx in range(data_train.review.shape[0]):
    text = BeautifulSoup(data_train.review[idx], "lxml")  #转化为beautifulsoup形式，用lxml解析
    texts.append(clean_str(text.get_text().encode('ascii','ignore').decode('ascii'))) #过滤非ascii字符
    labels.append(data_train.sentiment[idx]) #提取训练集中的情感标签

#GLOVE_DIR = "/data/mpk"
embeddings_index = {} #创建空字典
f = open(os.path.join('glove.6B.100d.txt'),'r',encoding='UTF-8')  #打开词向量f
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', len(texts))
print('Shape of label tensor:', len(labels))

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)#将数据序列化，并统一长度（这里统一句子长度为1000，多的截断，少的补0


indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]   #随机打乱数据，并将数据切分为训练集和验证集（切分比例8:2）


embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
print ('Length of embedding_matrix:', embedding_matrix.shape[0])
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            mask_zero=False,
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)  #未登录词（OOV问题）采取的是随机初始化向量，词向量不可训练。


print('Traing and validation set number of positive and negative reviews')
print (y_train.sum(axis=0))
print (y_val.sum(axis=0))

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
dense_1 = Dense(100,activation='tanh')(embedded_sequences)
max_pooling = GlobalMaxPooling1D()(dense_1)
dense_2 = Dense(2, activation='softmax')(max_pooling)

model = Model(sequence_input, dense_2)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=10, batch_size=50)
