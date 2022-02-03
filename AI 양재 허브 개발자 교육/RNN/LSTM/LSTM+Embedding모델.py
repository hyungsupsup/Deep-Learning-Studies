#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## LSTM 모델을 이용한 영화리뷰 분류 분석

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.datasets import imdb

### Step 1-1. Input tensor 와 Target tensor 준비(훈련데이터)
* IMDB 영화 리뷰 데이터 down

(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=10000)

### Step 1-2. 입력 데이터의 전처리 
* LSTM 모델에 데이터를 입력하기 위해 시퀀스 데이터의 길이를 통일

from tensorflow.keras.preprocessing import sequence

input_train = sequence.pad_sequences(input_train, 800)    # pad_sequences: Pads sequences to the same length
input_test = sequence.pad_sequences(input_test, 800)

input_train.shape, input_test.shape

### Step2. LSTM 모델 디자인

from tensorflow.keras import models, layers
model = models.Sequential()

# embedding layer: 32차원, hidden layer : LSTM 1개[32]
model.add(layers.Embedding(input_dim = 10000 , output_dim = 32,input_length = 800))
model.add(layers.LSTM(units= 32))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.summary()  #32개의 feature를 가지는 데이터가 32개의 퍼셉트론을 가지는 LSTM으로 전달됨

forget = 32*32*1 + 32*32*1 + 1*32  
input = 32*32*2 + 32*32*2 + 2*32
output = 32*32*1 + 32*32*1 + 1*32
forget + input + output

### Step 3. 모델의 학습 정보 설정

# loss : binary crossentropy/ optimizer : rmsprop/ metric : accuracy
model.compile(loss = 'binary_crossentropy',
              optimizer = 'rmsprop',
              metrics = 'accuracy')

### Step 4. 모델에 input, target 데이터 연결 후 학습

# batch size : 128, epochs : 10, validation data set percent : 20%
history = model.fit(x =input_train, y =y_train,
          batch_size = 128,
          epochs = 10,
          validation_split = 0.2)

### 학습과정의 시각화 및 성능 테스트

# 학습과정 시각화를 위한 정보 추출
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# 차트의 x 축을 위한 epoch 정보 생성
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
loss, accuracy = model.evaluate(input_test, y_test)

