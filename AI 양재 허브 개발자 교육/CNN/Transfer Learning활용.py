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

from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(rescale = 1./255, validation_split = 0.2)

test_datagen = ImageDataGenerator(rescale = 1./255)

* .flow_from_directory() 메서드를 이용하여 학습데이터와 검증데이터를 위한 DirectoryIterator 객체 생성


train_generator = train_datagen.flow_from_directory(
    train_dir, 
    target_size = (224,224),
    batch_size = 20,
    shuffle = True,
    class_mode = 'categorical',
    subset = 'training',
    seed = 7    #데이터가 항상 똑같은 순서로 뒤섞여있게 함
)

validation_generator = train_datagen.flow_from_directory(
    train_dir, 
    target_size = (224,224),
    batch_size = 20,
    shuffle = True,
    class_mode = 'categorical',
    subset = 'validation',
    seed = 7
)

test_generator = test_datagen.flow_from_directory(
    test_dir, 
    target_size = (224,224),
    batch_size = 20,
    class_mode = 'categorical'
)

### Step 2. VGG16을 Backbone 으로 하는 모델 디자인 및 학습 정보 설정

(1) Pre-trained 된 VGG16 모델 객체 생성
  * imagenet 데이터를 이용해 학습된 모델 객체 생성
  * classification layer 제외

from tensorflow.keras.applications import VGG16

conve_base = VGG16(
    weights = 'imagenet',
    include_top = False,
    input_shape = (224,224,3)
)

conve_base.summary()

(2) VGG16 Backbone 모델에 classification layer 추가

model = keras.Sequential()

model.add(conve_base)
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=256, activation='relu'))  #hidden layer 
model.add(keras.layers.Dense(units=3, activation='softmax'))

(3) VGG16 Backbone 모델의 가중치 동결(학습대상 가중치에서 제외)

* VGG16 Backbone 모델의 가중치 동결 및 동결 후 학습대상 파라미터 개수 출력

len(model.trainable_weights)   #len(): Return the number of items in a container

model.layers[0].trainable

model.summary()

from tensorflow.keras.utils import plot_model 
plot_model(model, show_shapes= True )

(4) 학습을 위한 설정 정보 지정

model.compile(loss= 'categorical_crossentropy',
              optimizer = keras.optimizers.RMSprop(learning_rate=2e-5),
              metrics = ['accuracy']
)

### Step 3. 모델에 데이터 generator 연결 후 학습 
  * model.fit() 이용하여 데이터 연결 및 학습시키기
  * 학습 과정은 history 변수에 저장

model.fit(train_generator, 
          epochs = 10,
          steps_per_epoch = len(train_generator),
          validation_data = validation_generator,
          validation_steps = len(validation_generator))

### Step 4. 테스트 데이터 셋을 통한 모델의 성능 평가

loss, accuracy = model.evaluate(test_generator)

#del model

