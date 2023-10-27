import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow import keras

model = []
epoch = 7
set_num = 3

def create_model():
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(150, 150, 3))

    model = keras.Sequential()
    model.add(conv_base)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    conv_base.trainable = False
    # print('conv_base layer 동결 후 훈련 가능 가중치 종류:', len(model.trainable_weights))
    return model

for i in range(set_num):
    model.append(create_model())

for i in range(set_num):
    model[i].load_weights("{}epoch {}.h5".format(epoch, i+1))

weights = []
for i in range(set_num):
    weights.append(model[i].get_weights())
print('weights 갯수 : ',len(weights))


new_weights_avg = list()
new_weights_median = list()
for weights_list_tuple in zip(*weights):
    new_weights_avg.append(
        np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
        # np.array([np.median(np.array(w), axis=0) for w in zip(*weights_list_tuple)])
    )
    new_weights_median.append(
        np.array([np.median(np.array(w), axis=0) for w in zip(*weights_list_tuple)])
    )

for i in range(3):
    weights[i] = np.array(weights[i], dtype=object)
    print(i)
    print(weights[i][-3:-2][0][:5])
    print(weights[i][-4:-3][0][0][:5])
    print("*************************************************************************************")

print("평균 weights:")
print(new_weights_avg[-3:-2][0][:5])
print(new_weights_avg[-4:-3][0][0][:5])
print("중간값 weights:")
print(new_weights_median[-3:-2][0][:5])
print(new_weights_median[-4:-3][0][0][:5])