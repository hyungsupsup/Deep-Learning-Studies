##### Copyright 2020 The TensorFlow Authors.

#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Keras 모델 저장 및 로드

<table class="tfo-notebook-buttons" align="left">
  <td><a target="_blank" href="https://www.tensorflow.org/guide/keras/save_and_serialize"><img src="https://www.tensorflow.org/images/tf_logo_32px.png">TensorFlow.org에서 보기</a></td>
  <td><a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs-l10n/blob/master/site/ko/guide/keras/save_and_serialize.ipynb" class=""><img src="https://www.tensorflow.org/images/colab_logo_32px.png">Google Colab에서 실행</a></td>
  <td><a target="_blank" href="https://github.com/tensorflow/docs-l10n/blob/master/site/ko/guide/keras/save_and_serialize.ipynb" class=""><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png">GitHub에서 소스 보기</a></td>
  <td><a href="https://storage.googleapis.com/tensorflow_docs/docs-l10n/site/ko/guide/keras/save_and_serialize.ipynb" class=""><img src="https://www.tensorflow.org/images/download_logo_32px.png">노트북 다운로드</a></td>
</table>

## 소개

Keras 모델은 다중 구성 요소로 이루어집니다.

- 모델에 포함된 레이어 및 레이어의 연결 방법을 지정하는 아키텍처 또는 구성
- 가중치 값의 집합("모델의 상태")
- 옵티마이저(모델을 컴파일하여 정의)

Keras API를 사용하면 이러한 조각을 한 번에 디스크에 저장하거나 선택적으로 일부만 저장할 수 있습니다.

- TensorFlow SavedModel 형식(또는 이전 Keras H5 형식)으로 모든 것을 단일 아카이브에 저장합니다. 이것이 표준 관행입니다.
- 일반적으로 JSON 파일로 아키텍처 및 구성만 저장합니다.
- 가중치 값만 저장합니다. 이것은 일반적으로 모델을 훈련할 때 사용됩니다.

언제 사용해야 하는지, 어떻게 동작하는 것인지 각각 살펴봅시다.

## 저장 및 로딩에 대한 짧은 답변

다음은 이 가이드를 읽는데 10초 밖에 없는 경우 알아야 할 사항입니다.

**Keras 모델 저장하기**

```python
model = ...  # Get model (Sequential, Functional Model, or Model subclass) model.save('path/to/location')
```

**모델을 다시 로딩하기**

```python
from tensorflow import keras model = keras.models.load_model('path/to/location')
```

이제 세부 사항을 확인해봅시다.

## 설정

import numpy as np
import tensorflow as tf
from tensorflow import keras

## 전체 모델 저장 및 로딩

전체 모델을 단일 아티팩트로 저장할 수 있습니다. 다음을 포함합니다.

- 모델의 아키텍처 및 구성
- 훈련 중에 학습된 모델의 가중치 값
- 모델의 컴파일 정보(`compile()`이 호출된 경우)
- 존재하는 옵티마이저와 그 상태(훈련을 중단한 곳에서 다시 시작할 수 있게 해줌)

#### APIs

- `model.save()` 또는 `tf.keras.models.save_model()`
- `tf.keras.models.load_model()`

전체 모델을 디스크에 저장하는 데 사용할 수 있는 두 형식은 **TensorFlow SavedModel 형식**과 **이전 Keras H5 형식**입니다. 권장하는 형식은 SavedModel입니다. 이는 `model.save()`를 사용할 때의 기본값입니다.

다음을 통해 H5 형식으로 전환할 수 있습니다.

- `save_format='h5'`를 `save()`로 전달합니다.
- `.h5` 또는 `.keras`로 끝나는 파일명을 `save()`로 전달합니다.

### SavedModel 형식

**예제:**

def get_model():
    # Create a simple model.
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


model = get_model()

# Train the model.
test_input = np.random.random((128, 32))
test_target = np.random.random((128, 1))
model.fit(test_input, test_target)

# Calling `save('my_model')` creates a SavedModel folder `my_model`.
model.save("my_model")

# It can be used to reconstruct the model identically.
reconstructed_model = keras.models.load_model("my_model")

# Let's check:
np.testing.assert_allclose(
    model.predict(test_input), reconstructed_model.predict(test_input)
)

# The reconstructed model is already compiled and has retained the optimizer
# state, so training can resume:
reconstructed_model.fit(test_input, test_target)

#### SavedModel이 포함하는 것

`model.save('my_model')`을 호출하면 다음을 포함하는  `my_model` 폴더를 생성합니다.

!ls my_model

### Keras H5 형식

Keras는 또한 모델의 아키텍처, 가중치 값 및 `compile()` 정보가 포함된 단일 HDF5 파일 저장을 지원합니다. SavedModel에 대한 가벼운 대안입니다.

**예제:**

model = get_model()

# Train the model.
test_input = np.random.random((128, 32))
test_target = np.random.random((128, 1))
model.fit(test_input, test_target)

# Calling `save('my_model.h5')` creates a h5 file `my_model.h5`.
model.save("my_h5_model.h5")   #.h5를 붙인다

# It can be used to reconstruct the model identically.
reconstructed_model = keras.models.load_model("my_h5_model.h5")

# Let's check:
np.testing.assert_allclose(
    model.predict(test_input), reconstructed_model.predict(test_input)
)

# The reconstructed model is already compiled and has retained the optimizer
# state, so training can resume:
reconstructed_model.fit(test_input, test_target)

## 모델의 가중치 값만 저장 및 로딩

모델의 가중치 값만 저장하고 로드하도록 선택할 수 있습니다. 다음과 같은 경우에 유용할 수 있습니다.

- 추론을 위한 모델만 필요합니다. 이 경우 훈련을 다시 시작할 필요가 없으므로 컴파일 정보나 옵티마이저 상태가 필요하지 않습니다.
- 전이 학습을 수행하고 있습니다. 이 경우 이전 모델의 상태를 재사용하는 새 모델을 훈련하므로 이전 모델의 컴파일 정보가 필요하지 않습니다.

### 인메모리 가중치 전이를 위한 API

`get_weights` 및 `set_weights`를 사용하여 다른 객체 간에 가중치를 복사할 수 있습니다.

- `tf.keras.layers.Layer.get_weights()`: numpy 배열의 리스트를 반환합니다.
- `tf.keras.layers.Layer.set_weights()`: `weights` 인수 내 값으로 모델의 가중치를 설정합니다.

다음은 예제입니다.

***메모리에서 레이어 간 가중치 전이하기***

def create_layer():
    layer = keras.layers.Dense(64, activation="relu", name="dense_2")
    layer.build((None, 784))
    return layer

layer_1 = create_layer()
layer_2 = create_layer()

layer_1.get_weights()

layer_2.get_weights()

# Copy weights from layer 2 to layer 1
layer_1.set_weights(layer_2.get_weights())

layer_1.get_weights()

***메모리에서 호환 가능한 아키텍처를 사용하여 모델 간 가중치 전이하기***

# Create a simple functional model
inputs = keras.Input(shape=(784,), name="digits")
x = keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = keras.layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = keras.layers.Dense(10, name="predictions")(x)
functional_model = keras.Model(inputs=inputs, outputs=outputs, name="3_layer_mlp")

# Define a subclassed model with the same architecture
class SubclassedModel(keras.Model):
    def __init__(self, output_dim, name=None):
        super(SubclassedModel, self).__init__(name=name)
        self.output_dim = output_dim
        self.dense_1 = keras.layers.Dense(64, activation="relu", name="dense_1")
        self.dense_2 = keras.layers.Dense(64, activation="relu", name="dense_2")
        self.dense_3 = keras.layers.Dense(output_dim, name="predictions")

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return x

    def get_config(self):
        return {"output_dim": self.output_dim, "name": self.name}

# Call the subclassed model once to create the weights.
subclass_model = SubclassedModel(output_dim=10)
subclass_model(tf.ones((1, 784)))

len(subclass_model.weights)

functional_model.layers[1].weights[0]

subclass_model.layers[0].weights[0]

# Copy weights from functional_model to subclassed_model.
subclass_model.set_weights(functional_model.get_weights())

***상태 비저장 레이어의 경우***

상태 비저장 레이어는 순서 또는 가중치 수를 변경하지 않기 때문에 상태 비저장 레이어가 남거나 없더라도 모델은 호환 가능한 아키텍처를 가질 수 있습니다.

inputs = keras.Input(shape=(784,), name="digits")
x = keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = keras.layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = keras.layers.Dense(10, name="predictions")(x)
functional_model = keras.Model(inputs=inputs, outputs=outputs, name="3_layer_mlp")

inputs = keras.Input(shape=(784,), name="digits")
x = keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = keras.layers.Dense(64, activation="relu", name="dense_2")(x)

# Add a dropout layer, which does not contain any weights.
x = keras.layers.Dropout(0.5)(x)
outputs = keras.layers.Dense(10, name="predictions")(x)
functional_model_with_dropout = keras.Model(
    inputs=inputs, outputs=outputs, name="3_layer_mlp"
)

functional_model.get_weights()

functional_model_with_dropout.set_weights(functional_model.get_weights())

functional_model_with_dropout.get_weights()

### 디스크에 가중치를 저장하고 다시 로딩하기 위한 API

다음 형식으로 `model.save_weights`를 호출하여 디스크에 가중치를 저장할 수 있습니다.

- TensorFlow Checkpoint
- HDF5

`model.save_weights`의 기본 형식은 TensorFlow 체크포인트입니다. 저장 형식을 지정하는 두 가지 방법이 있습니다.

1. `save_format` 인수: `save_format="tf"` 또는 `save_format="h5"`에 값을 설정합니다.
2. `path` 인수: 경로가 `.h5` 또는 `.hdf5`로 끝나면 HDF5 형식이 사용됩니다. `save_format`을 설정하지 않으면 다른 접미어의 경우 TensorFlow 체크포인트로 결과가 발생합니다.

인메모리 numpy 배열로 가중치를 검색하는 옵션도 있습니다. 각 API에는 장단점이 있으며 아래에서 자세히 설명합니다.

### TF Checkpoint 형식

**예제:**

# Runnable example
sequential_model = keras.Sequential(
    [
        keras.Input(shape=(784,), name="digits"),
        keras.layers.Dense(64, activation="relu", name="dense_1"),
        keras.layers.Dense(64, activation="relu", name="dense_2"),
        keras.layers.Dense(10, name="predictions"),
    ]
)

sequential_model.layers[1].weights  #두번째 레이어의 가중치 정보

sequential_model.save_weights('ckpt')

# Runnable example
restore_model = keras.Sequential(
    [
        keras.Input(shape=(784,), name="digits"),
        keras.layers.Dense(64, activation="relu", name="dense_1"),
        keras.layers.Dense(64, activation="relu", name="dense_2"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, name="predictions"),
    ]
)

restore_model.layers[1].weights

load_status = restore_model.load_weights('ckpt')

# `assert_consumed` can be used as validation that all variable values have been
# restored from the checkpoint. See `tf.train.Checkpoint.restore` for other
# methods in the Status object.

load_status.assert_consumed()

#### 형식 세부 사항

TensorFlow Checkpoint 형식은 객체 속성명을 사용하여 가중치를 저장하고 복원합니다. 예를 들어, `tf.keras.layers.Dense` 레이어를 고려해 봅시다. 레이어에는 `dense.kernel`과 `dense.bias` 두 가지 가중치가 있습니다. 레이어가 `tf` 형식으로 저장되면 결과 체크포인트에는 `"kernel"` 및 `"bias"`와 해당 가중치 값이 포함됩니다.

#### 전이 학습 예제

기본적으로 두 모델이 동일한 아키텍처를 갖는 한 동일한 CheckPoint를 공유할 수 있습니다.

**예제:**

inputs = keras.Input(shape=(784,), name="digits")
x = keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = keras.layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = keras.layers.Dense(10, name="predictions")(x)
functional_model = keras.Model(inputs=inputs, outputs=outputs, name="3_layer_mlp")

# Extract a portion of the functional model defined in the Setup section.
# The following lines produce a new model that excludes the final output
# layer of the functional model.
pretrained = keras.Model(
    inputs=inputs,
    outputs=functional_model.layers[-2].output
)

pretrained.summary()

# Randomly assign "trained" weights.
for w in pretrained.weights:
    w.assign(tf.random.normal(w.shape))

pretrained.layers[1].weights

pretrained.save_weights('pretrained_ckp')

# Assume this is a separate program where only 'pretrained_ckpt' exists.
# Create a new functional model with a different output dimension.
inputs = keras.Input(shape=(784,), name="digits")
x = keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = keras.layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = keras.layers.Dense(5, name="predictions")(x)
model = keras.Model(inputs=inputs, outputs=outputs, name="new_model")

# Load the weights from pretrained_ckpt into model.
model.load_weights('pretrained_ckp')

model.layers[1].weights

# Check that all of the pretrained weights have been loaded.
for a, b in zip(pretrained.weights, model.weights):
    np.testing.assert_allclose(a.numpy(), b.numpy())

model.summary()

# Example 2: Sequential model
# Recreate the pretrained model, and load the saved weights.
model = keras.Sequential(
    [
      keras.layers.Input(shape=(784,), name="digits"),
      keras.layers.Dense(64, activation="relu", name="dense_1"),
      keras.layers.Dense(64, activation="relu", name="dense_2"),
      keras.layers.Dense(5, name="predictions")
     ]
)

# Sequential example:
model.load_weights('pretrained_ckp')

# Check that all of the pretrained weights have been loaded.
for a, b in zip(pretrained.weights, model.weights):
    np.testing.assert_allclose(a.numpy(), b.numpy())

일반적으로 모델을 빌드할 때 동일한 API를 사용하는 것이 좋습니다. Sequential 및 Functional 또는 Functional 및 서브 클래스 등 간에 전환하는 경우, 항상 사전 훈련된 모델을 다시 빌드하고 사전 훈련된 가중치를 해당 모델에 로드합니다.

### HDF5 format

HDF5 형식에는 레이어 이름별로 그룹화된 가중치가 포함됩니다. 가중치는 훈련 가능한 가중치 목록을 훈련 불가능한 가중치 목록(`layer.weights`와 동일)에 연결하여 정렬된 목록입니다. 따라서 모델이 체크포인트에 저장된 것과 동일한 레이어 및 훈련 가능한 상태를 갖는 경우 hdf5 체크포인트을 사용할 수 있습니다.

**예제:**

# Runnable example
sequential_model = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape=(784,), name="digits"),
        keras.layers.Dense(64, activation="relu", name="dense_1"),
        keras.layers.Dense(64, activation="relu", name="dense_2"),
        keras.layers.Dense(10, name="predictions"),
    ]
)

sequential_model.save_weights('weights.h5')

# Runnable example
sequential_restore_model = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape=(784,), name="digits"),
        keras.layers.Dense(64, activation="relu", name="dense_1"),
        keras.layers.Dense(64, activation="relu", name="dense_2"),
        keras.layers.Dense(10, name="predictions"),
    ]
)

sequential_restore_model.load_weights('weights.h5')

sequential_model.layers[0].weights

sequential_restore_model.layers[0].weights

#### 전이 학습 예제

HDF5에서 사전 훈련된 가중치를 로딩할 때는 가중치를 기존 체크포인트 모델에 로드한 다음 원하는 가중치/레이어를 새 모델로 추출하는 것이 좋습니다.

**예제:**

def create_functional_model():
    inputs = keras.Input(shape=(784,), name="digits")
    x = keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = keras.layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = keras.layers.Dense(10, name="predictions")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="3_layer_mlp")

functional_model = create_functional_model()

functional_model.save_weights('pretrained_weight.h5')

functional_model.layers[1].weights

# In a separate program:
functional_restore_model = create_functional_model()

functional_restore_model.load_weights('pretrained_weight.h5')

functional_restore_model.layers[1].weights

# Create a new model by extracting layers from the original model:
extracted_layers = functional_restore_model.layers[:-1]
extracted_layers.append(keras.layers.Dense(5, name="predictions"))

extracted_layers

model = keras.Sequential(extracted_layers)

model.summary()

