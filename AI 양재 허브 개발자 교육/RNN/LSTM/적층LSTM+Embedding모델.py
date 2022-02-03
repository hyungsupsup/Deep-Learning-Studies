#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## 적층 LSTM 모델을 이용한 영화리뷰 분류 분석

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.datasets import imdb

### Step 1-1. Input tensor 와 Target tensor 준비(훈련데이터)
* IMDB 영화 리뷰 데이터 down

(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=10000)

### Step 1-2. 입력 데이터의 전처리 
* LSTM 모델에 데이터를 입력하기 위해 시퀀스 데이터의 길이를 통일

from tensorflow.keras.preprocessing import sequence

input_train = sequence.pad_sequences(input_train, 800)
input_test = sequence.pad_sequences(input_test, 800)

input_train.shape, input_test.shape

### Step2. 적층 LSTM 모델 디자인

from tensorflow.keras import models, layers
model = models.Sequential()

# embedding layer: 32차원, hidden layer : LSTM 2개[32, 16], dropout rate : 0.5
model.add(layers.Embedding(input_dim = 10000, output_dim = 32, input_length = 800))

model.add(layers.LSTM(units =32, return_sequences=True))    #return_sequences=True로 해야 적층 LSTM

model.add(layers.GRU(units = 16))  # GRU는 LSTM과 비슷한 역할을 하지만, 더 간단한 구조로 이루어져 있어서 계산상으로 효율적

model.add(layers.Dropout(0.5))  # 모델의 과적합 문제는 정규화(regularization) 방법을 주로 사용해 해결하는데, Dropout 함수는 정규화의 방식중 하나인 드롭아웃을 쉽게 구현해주는 함수이다(즉, 입력 데이터에 드롭아웃이 적용됨).
# 과적합을 방지하기 위해 무작위로 특정 노드(입력값)를 0으로 만든다. 물론 드롭아웃은 학습시에만 적용되어 모델 정규화를 위해 사용되어야 하며 테스트시에는 적용되서는 안된다.

model.add(layers.Dense(1, activation= 'sigmoid'))        

model.summary()

# loss : binary crossentropy/ optimizer : rmsprop/ metric : accuracy
model.compile(optimizer= 'rmsprop',
              loss = 'binary_crossentropy',
              metrics = 'accuracy')

# batch size : 128, epochs : 10, validation data set percent : 20%
history = model.fit(x = input_train, y= y_train,
          batch_size=128,
          epochs= 10,
          validation_split =0.2)

# 학습과정 시각화를 위한 정보 추출
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# 정합도 정보 시각화
plt.plot(epochs, acc, label='Training acc')
plt.plot(epochs, val_acc, label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

# loss 정보 시각화
plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# 테스트 데이터 셋을 통한 성능 측정

