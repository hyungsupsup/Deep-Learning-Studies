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

features_considered = ['p (mbar)', 'T (degC)', 'rho (g/m**3)']

features = df[features_considered]
features.index = df['Date Time']
features.head()

features.plot(subplots=True)

data_url = 'https://codepresso-online-platform-public.s3.ap-northeast-2.amazonaws.com/learning-resourse/Tensorflow+2.0+%EB%94%A5%EB%9F%AC%EB%8B%9D+%EC%99%84%EB%B2%BD+%EA%B0%80%EC%9D%B4%EB%93%9C/multivariate-temperature-codepresso.npz'

# requests 라이브러리를 이용해 데이터 다운로드
response = requests.get(data_url)
response.raise_for_status()

# 다운로드 받은 데이터를 읽어 들여 Input tensor 와 Target tensor 준비
with np.load(io.BytesIO(response.content)) as tempt_codepresso_data:
#with np.load(io.BytesIO(response.content)) as tempt_codepresso_data:
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
  * 관찰 점 당 feature의 개수 : 3 개(p (mbar), T (degC), rho (g/m**3))
  * 예측 시퀀스 : 0(관찰기간 바로 뒤 시퀀스) 

train_features.shape, train_labels.shape

test_features.shape, test_labels.shape

def show_plot(plot_data, title):
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
      for feature_index in range(plot_data[i].shape[1]):
        plt.plot(time_steps, plot_data[i][ :, feature_index], marker[i], label=labels[i])
  plt.legend()
  plt.xlim([time_steps[0], (future+5)*2])
  plt.xlabel('Time-Step')
  return plt

show_plot([train_features[0], train_labels[0]], 'Sample Example')

mean = train_features.reshape((-1,3)).mean(axis=0)
std = train_features.reshape((-1,3)).std(axis=0)

mean, std   # (p (mbar), T (degC), rho (g/m**3))

train_features = (train_features-mean)/std   #표준화 시켜줌
test_features = (test_features-mean)/std

train_labels = (train_labels-mean[1])/std[1]    #온도  mean[1], std[1]
test_labels = (test_labels-mean[1])/std[1]      #온도

train_labels

show_plot ([train_features[0], train_labels[0]] , '' )

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.SimpleRNN(units= 32, input_shape=(20,3)))   
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer = 'adam',
              loss = 'mse',
              metrics= 'mae')

wx = 3*32
wh = 32*32 #context를 분석하는 weight의 개수 
b = 1*32
wx+wh+b


model.summary()

history = model.fit(
      train_features, y = train_labels,
      batch_size = 256,
      epochs = 10,
      validation_split = 0.2)

test_loss, test_mae = model.evaluate(test_features, test_labels)

def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  plt.show()

plot_train_history(history, 'title')

for x, y in zip(test_features[:3], test_labels[:3]):
  plot = show_plot([x[:,1].reshape(20,1), y, model.predict(x.reshape(1,20,3))[0]], 'Multivar RNN model')
  plot.show()

def denormalize(data):
    de_nor_data = data*std[1] + mean[1]
    return de_nor_data

y_pred = denormalize(model.predict(test_features)).reshape(-1,)

y_true = denormalize(test_labels)

denor_test_mae = np.mean(np.absolute(y_true - y_pred))

denor_test_mae

#적층 RNN 모델을 이용한 분석------>데이터의 복잡도가 높을 때 활용

stacked_model = tf.keras.Sequential()

from tensorflow.keras import layers

stacked_model.add(layers.SimpleRNN(32, input_shape = (20,3),
                                   return_sequences = True))  #매 시퀀스마다 아웃풋 레이어를 뱉어냄
stacked_model.add(layers.SimpleRNN(16))
stacked_model.add(layers.Dense(1))

stacked_model.summary()  #simple_rnn_2-->퍼셉트론 하나 당 하나의 아웃풋이 나옴(return_sequences = False)

model.compile(loss='mse',
              optimizer='adam',
              metrics=['mae'])

history = model.fit(
      train_features, y = train_labels,
      batch_size = 256,
      epochs = 20,
      validation_split = 0.2)

test_loss, test_mae = model.evaluate(test_features, test_labels)

