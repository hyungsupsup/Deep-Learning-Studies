#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow import keras

## Data Augmention 을 통한 가위-바위-보 데이터 셋 분류 성능 개선

### Step 1. Input tensor 와 Target tensor 준비(훈련데이터)

(1) 가위-바위-보 데이터셋 다운로드

url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip'

path_to_zip = keras.utils.get_file('rps.zip',
                                   origin=url,
                                   extract=True,
                                   cache_dir = '/content')

url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip'

path_to_zip = keras.utils.get_file('rps_test.zip',
                                   origin=url,
                                   extract=True,
                                   cache_dir = '/content')

(2) ImageDataGenerator를 이용해 이미지 파일을 load 하기 위한 경로 지정

train_dir = '/content/datasets/rps'
test_dir = '/content/datasets/rps-test-set'

(3) ImageDataGenerator 객체 생성  
* 객체 생성 시 rescale 인자를 이용하여 텐서 내 원소의 범위를 [0 ~ 255] => [0 ~ 1] 로 ReScaling 진행
* Data Augmentation 을 위한 정보들 설정

# rotation : 40 
# width : 0.2
# height : 0.2
# shear: 0.2
# zoom: 0.2
# horizontal: True

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale = 1./255, validation_split = 0.2)

test_datagen = ImageDataGenerator(rescale = 1./255)

* .flow_from_directory() 메서드를 이용하여 학습데이터와 검증데이터를 위한 DirectoryIterator 객체 생성


train_generator = train_datagen.flow_from_directory(
    train_dir, 
    target_size = (150,150),
    batch_size = 32,
    shuffle = True,
    class_mode = 'categorical',
    subset = 'training',
    seed = 7    #데이터가 항상 똑같은 순서로 뒤섞여있게 함
)

validation_generator = validation_datagen.flow_from_directory(
    train_dir, 
    target_size = (150,150),
    batch_size = 32,
    shuffle = True,
    class_mode = 'categorical',
    subset = 'validation'
)

test_generator = test_datagen.flow_from_directory(
    test_dir, 
    target_size = (150,150),
    batch_size = 32,
    class_mode = 'categorical'
)

### Step 2. CNN 모델 디자인 및 학습 정보 설정

(1) CNN 모델 구성

from tensorflow.keras import models, layers
model = models.Sequential()

# 1st Conv2D  -filter_cnt : 32  -kerner_size : (3,3)  -relu
model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(150,150,3)))

# 1st max pooling  -pool_size = (2,2)
model.add(layers.MaxPool2D(pool_size=(2,2)))

#2nd Conv2D - filter_cnt : 64 -kerner_size: (3,3) -relu
model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))

#2nd max pooling - pool_size = (2,2) 
model.add(layers.MaxPool2D(pool_size=(2,2)))

#3rd Conv2D - filter_cnt : 128 -kerner_size : (3,3) -relu
model.add(layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'))

#2nd max pooling - pool_size = (2,2) 
model.add(layers.MaxPool2D(pool_size=(2,2)))

#3rd Conv2D - filter_cnt : 128 -kerner_size : (3,3) -relu
model.add(layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'))

#2nd max pooling - pool_size = (2,2) 
model.add(layers.MaxPool2D(pool_size=(2,2)))

#Flatten
model.add(layers.Flatten())

#classification module
model.add(layers.Dense(units=512, activation='relu'))  #hidden layer
model.add(layers.Dense(units=3, activation='sigmoid'))  #output layer
#model.add(layers.Dense(units=10)) # softmax를 뺀 output layer

(2) 학습을 위한 설정 정보 지정

model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits = True),
              optimizer=  'rmsprop',
              metrics = ['accuracy'])

### Step 3. 모델에 데이터 generator 연결 후 학습 
  * model.fit() 이용하여 데이터 연결 및 학습시키기
  * 학습 과정은 history 변수에 저장

history = model.fit(train_generator,   #x인자 y인자 대신 generator를 넣어줌 
          epochs = 10,
          steps_per_epoch = len(train_generator),
          validation_data = test_generator,
          validation_steps = len(test_generator))

### Step 4. 테스트 데이터 셋을 통한 모델의 성능 평가

loss, accuracy = model.evaluate(test_generator)

