#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow import keras

## Transfer Learning 을 통한 가위-바위-보 분류 데이터 셋 분류 성능 개선

### Step 1. Input tensor 와 Target tensor 준비(훈련데이터)

(1) 가위-바위-보 데이터셋 다운로드

url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip'

path_to_zip = keras.utils.get_file('rps.zip',
                                   origin=url,
                                   extract=True,
                                   cache_dir='/content')

url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip'

path_to_zip = keras.utils.get_file('rps_test.zip',
                                   origin=url,
                                   extract=True,
                                   cache_dir='/content')

(2) ImageDataGenerator를 이용해 이미지 파일을 load 하기 위한 경로 지정

train_dir = '/content/datasets/rps'
test_dir = '/content/datasets/rps-test-set'

(3) ImageDataGenerator 객체 생성  
* 객체 생성 시 rescale 인자를 이용하여 텐서 내 원소의 범위를 [0 ~ 255] => [0 ~ 1] 로 ReScaling 진행

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 모든 이미지를 1/255로 스케일을 조정합니다
train_datagen = ImageDataGenerator(rescale=1./255,
                                   validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

* .flow_from_directory() 메서드를 이용하여 학습데이터와 검증데이터를 위한 DirectoryIterator 객체 생성

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=20,
        shuffle=True,
        class_mode='categorical',
        subset='training',
        seed=7)

validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=20,
        shuffle=True,
        class_mode='categorical',
        subset='validation',
        seed=7)

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=20,
        class_mode='categorical')

### Step 2. VGG16을 Backbone 으로 하는 모델 디자인 및 학습 정보 설정

(1) Pre-trained 된 VGG16 모델 객체 생성
  * imagenet 데이터를 이용해 학습된 모델 객체 생성
  * classification layer 제외

from tensorflow.keras.applications import VGG16

conv_base = VGG16(include_top=False,
                  weights='imagenet',
                  input_shape=(224, 224, 3))

conv_base.summary()

(2) VGG16 Backbone 모델에 classification layer 추가

model = keras.Sequential()

model.add(conv_base)
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))

model.summary()

(3) VGG16 Backbone 모델의 가중치 동결(학습대상 가중치에서 제외)

* VGG16 Backbone 모델의 가중치 중 마지막 Conv2D Layer(block5_conv3) 를 제외한 layer 들의 가중치 동결

len(model.trainable_weights)

> * 방법1. conv_base 객체에서 .layers 속성 정보를 이용하여 모델에 추가되어 있는 layer 객체 들에 접근 하여 loop 돌면서 name 이 `block5_conv3` 인 layer를 제외 하고 동결 처리



for layer in conv_base.layers:
  if layer.name == 'block5_conv3':    
    continue
  layer.trainable=False                #마지막 layer인 'block5_conv3'인 layer 제외하고 전부 동결시킴

len(model.trainable_weights)

conv_base.get_layer('block5_conv3')   

#conv_base.trainable = True

#len(model.trainable_weights)

> * 방법2. conv_base 객체에서 .layers 속성 정보를 이용하여 모델에 추가되어 있는 layer 객체 들에 접근 하여 loop 돌면서 동결 처리 후 동결에서 제외 하고자 하는 layer 선택하여 동결 해제



len(model.trainable_weights)

for layer in conv_base.layers:
  layer.trainable=False     #상위 layer들을 False로 바꿔주면 그것의 하위모델도 False로 바뀜

len(model.trainable_weights)

conv_base.layers[-2].trainable = True

len(model.trainable_weights)

model.summary()

(4) 학습을 위한 설정 정보 지정

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop(learning_rate=2e-5),
              metrics=['accuracy'])

### Step 3. 모델에 데이터 generator 연결 후 학습 
  * model.fit() 이용하여 데이터 연결 및 학습시키기
  * 학습 과정은 history 변수에 저장

history = model.fit(
      train_generator,
      steps_per_epoch=len(train_generator),
      epochs=30,
      validation_data=validation_generator,
      validation_steps=len(validation_generator))

### Step 4. 테스트 데이터 셋을 통한 모델의 성능 평가

loss, accuracy = model.evaluate(test_generator)

