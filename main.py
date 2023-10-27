import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import h5py
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

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

model = []
epoch = 10
set_num = 100
for i in range(set_num):
    model.append(create_model())

new_model = create_model()

for i in range(set_num):
    model[i].load_weights("{}epoch {}.h5".format(epoch, i+1))

weights = []
for i in range(set_num):
    weights.append(model[i].get_weights())
# print('weights 갯수 : ',len(weights))


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

#데이터셋 전처리
url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = keras.utils.get_file('cats_and_dogs.zip', origin=url, extract=True)
data_path = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

base_dir = data_path
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        shuffle=True,
        class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

#모델 evaluate
new_model.set_weights(new_weights_avg)
new_model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(lr=2e-5), metrics=['accuracy'])
avg_test_loss, avg_test_acc = new_model.evaluate(validation_generator)

new_model.set_weights(new_weights_median)
new_model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(lr=2e-5), metrics=['accuracy'])
test_loss, test_acc = new_model.evaluate(validation_generator)

print('avg 테스트 정확도,loss:', avg_test_acc, avg_test_loss)
print('median 테스트 정확도,loss:', test_acc, test_loss)