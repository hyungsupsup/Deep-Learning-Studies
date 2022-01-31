#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import requests
import io

## The weather dataset
- 막스 플랑크 연구소에 제공한 기후 관련 데이터  
  * 14 개의 특성 정보(air temperature, atmospheric pressure, and humidity, etc)
  * 2003년 부터 매 10분 마다 측정된 데이터  
    * 한 시간 동안 6 개의 관측치 존재
    * 하루에 144 (6x24) 관측치가 포함
  * 효율적인 학습을 위해 2009 ~ 2016 데이터를 이용하여 학습 예정


zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path) #파일받아오기

df = pd.read_csv(csv_path)

df.head()

#단순 회귀 분석
uni_data = df['T (degC)']
uni_data.index = df['Date Time']
uni_data.head()

uni_data.plot(subplots=True)

data_url = 'https://codepresso-online-platform-public.s3.ap-northeast-2.amazonaws.com/learning-resourse/Tensorflow+2.0+%EB%94%A5%EB%9F%AC%EB%8B%9D+%EC%99%84%EB%B2%BD+%EA%B0%80%EC%9D%B4%EB%93%9C/univariate-temperature-codepresso.npz'

# requests 라이브러리를 이용해 데이터 다운로드
response = requests.get(data_url)
response.raise_for_status()

# 다운로드 받은 데이터를 읽어 들여 Input tensor 와 Target tensor 준비
with np.load(io.BytesIO(response.content)) as tempt_codepresso_data:
  # 학습 이미지 데이터(np.ndarry, shape=(299980, 20, 1))
  train_features = tempt_codepresso_data['train_features']
  # 학습 라벨 데이터(np.ndarry, shape=(299980,))
  train_labels = tempt_codepresso_data['train_labels']
  
  # 테스트 이미지 데이터(np.ndarry, shape=(120531, 20, 1))
  test_features = tempt_codepresso_data['test_features']
  # 테스트 라벨 데이터(np.ndarry, shape=(120531,))
  test_labels = tempt_codepresso_data['test_labels']

* 학습을 위한 univariate-temperature-codepresso.npz 데이터 셋 구성
  * 관찰기간 : 20(200분)
  * 관찰 점 당 feature의 개수 : 1 개(T (degC))
  * 예측 시퀀스 : 0(관찰기간 바로 뒤 시퀀스) 

train_features.shape, test_features.shape  #20개의 sequence, 각 sequence는 1개의 원소를 가짐

train_labels.shape, test_labels.shape

def show_plot(plot_data, title):        #모델이 예측을 잘하고 있는지 검증하는 함수
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']
  time_steps = list(range(-plot_data[0].shape[0], 0))
  future = 0

  plt.title(title)
  for i, x in enumerate(plot_data):
    if i:
      plt.plot(future, plot_data[i], marker[i], markersize=10,
               label=labels[i])
    else:
      plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
  plt.legend()
  plt.xlim([time_steps[0], (future+5)*2])
  plt.xlabel('Time-Step')
  return plt

show_plot([train_features[0], train_labels[0]], 'Sample Example')   #첫번째 인자: 20개의 sequence를 가지고 있는 input 데이터

#del model

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.SimpleRNN(units= 8, input_shape=(20,1)))   #들어오는 데이터는 20개의 시퀀스를 가지고 각각의 시퀀스는 1개의 원소를 가짐,
#return sequences 값이 True 이므로 매 시퀀스마다 context 생성돼서 output 생성됨---> 총 20개의 output , 각각 처리해서 각각의 예측값을 뱉어냄
model.add(tf.keras.layers.Dense(1))

model.compile(loss='mse',
              optimizer='adam',
              metrics=['mae'])

model.summary()     #

# wx = 1*8
# wh = 8*8
# b = 1*8
# wx + wh + b = 80

history = model.fit(
      train_features, y = train_labels,
      batch_size = 256,
      epochs = 10,
      validation_split = 0.2)

mae = history.history['mae']
val_mae = history.history['val_mae']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(mae) +1)

plt.plot(epochs, mae, 'bo', label='Training mae')
plt.plot(epochs, val_mae, 'b', label='Validation mae')
plt.title('Training and validation mae')
plt.legend()

plt.show()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

test_loss, test_mae = model.evaluate(test_features, test_labels)

for x, y in zip(test_features[:3], test_labels[:3]):
  plot = show_plot([x, y, model.predict(x.reshape(1,20,1))[0]], 'Univar RNN model')
  plot.show()

