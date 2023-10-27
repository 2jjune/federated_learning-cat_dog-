import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

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

set_epoch = 10
set_num=100
model = []
for i in range(set_num):
    model.append(create_model())

for i in range(set_num):
    model[i].compile(loss='binary_crossentropy', optimizer=keras.optimizers.RMSprop(lr=2e-5), metrics=['accuracy'])

for i in range(len(model)):
    print()
    print(i,"*******************************")
    model[i].fit(train_generator, steps_per_epoch=len(train_generator), epochs=set_epoch, validation_data=validation_generator, validation_steps=len(validation_generator))
    model[i].save_weights('{}epoch {}.h5'.format(set_epoch, i+1))
