#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## RNN 모델을 이용한 영화리뷰 분류 분석

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.datasets import imdb

### Step 1-1. Input tensor 와 Target tensor 준비(훈련데이터)
* IMDB 영화 리뷰 데이터 down

# num_words=10000 : 데이터에서 등장 빈도 순위로 몇 번째에 해당하는 단어까지를 사용할 것인지를 의미
(train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=10000)       #imdb데이터는 토큰화, 라벨 인코딩 등의 전처리를 미리 해뒀음

type(train_x), type(train_x[0])  #둘의 타입 정보가 달라서 위에서 warning 이 뜸

train_x.shape , train_y.shape

print(train_x[0])  

#무슨 문장인지 궁금하므로 아래 라이브러리 활용 
word_idx = imdb.get_word_index()

idx_word = {idx +3 : word for word, idx in word_idx.items() }     
#주의할 점은 imdb.get_word_index()에 저장된 값에 +3을 해야 실제 맵핑되는 정수입니다. 이것은 IMDB 리뷰 데이터셋에서 정한 규칙입니다

#idx_word

idx_word[4]  #idx가 3이하면 error 발생

idx_word[0] = '<pad>'
idx_word[1] = '<sos>'
idx_word[2] = '<unk>'

tokens = [idx_word[idx] for idx in train_x[24999]]

print(tokens)

### Step 1-2. 입력 데이터의 전처리 
* RNN 모델에 데이터를 입력하기 위해 시퀀스 데이터의 길이를 통일

from tensorflow.keras.preprocessing.sequence import pad_sequences

train_x = pad_sequences(train_x, maxlen = 800)    #우리가 가지고 있는 문장을 특정 길이로 맞춰줌
test_x = pad_sequences(test_x, maxlen = 800)  

train_x.shape, test_x.shape    #800개의 시퀀스로 잘 잘려져 있음

## 

### Step2. RNN 모델 디자인

from tensorflow.keras import models, layers

model = models.Sequential()

#embedding layer: 32차원, hidden layer: RNN 1개[32개의 퍼셉트론 배치], activation: tanh
#embedding layer
#RNN
#FCL
#ouput layer

model.add(layers.Embedding(input_dim = 10000, output_dim =  32, input_length = 800))   #weight 덩어리를 만들어 놓음, 토큰의 개수, 변화시키고자 하는 원소의 개수
model.add(layers.SimpleRNN(units= 32))
model.add(layers.Dense(1, activation = 'sigmoid'))


model.summary()   # 10000토큰 * 32 = 320000개의 파라미터
#simple_rnn : 32개의 원소를 가지는 1D 파라미터

### Step 3. 모델의 학습 정보 설정

model.compile(loss = 'binary_crossentropy',
              optimizer = 'rmsprop',
              metrics = 'accuracy')

### Step 4. 모델에 input, target 데이터 연결 후 학습

history = model.fit(x = train_x, y = train_y,
          batch_size = 128,
          epochs = 10,
          validation_split = 0.2)

loss, accuracy = model.evaluate(test_x, test_y)

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
test_loss, test_accuracy = model.evaluate(x = test_x, y = test_y)

import numpy as np 

test_sequences = np.delete(test_x[0], np.argwhere(test_x[1] == 0))

print(test_sequences)

sentence = [idx_word.get(idx, "Nono") for idx in test_sequences]    #out of index 피하기 위해 get 사용--> 만약 존재하지않으면 Nono로 출력하게 함

print(sentence)

model.predict(test_x[1].reshape(1,-1))

