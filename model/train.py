import numpy as np

import csv
import os
import json

from keras.utils import to_categorical
from keras import metrics, optimizers
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D
from keras.preprocessing import image
from scipy.io import wavfile 
from feature import extract_spec
from keras import backend

backend.clear_session()

csvfile = open('ESC-50-master/meta/esc50.csv' ,'rb')
lines = csv.reader(csvfile)

data = []
label = []

for line in list(lines)[1:]:
    file_name = line[0]
    file_path = os.path.join('ESC-50-master/audio', file_name)
    if os.path.exists(file_path):
        sr, y = wavfile.read(file_path)
        mel_spec_power = extract_spec(y, sr)
        data.append(mel_spec_power)
        label.append(line[2])


n_samples, height, width = np.shape(data) 
data = np.reshape(data, (n_samples, height, width, 1))
label = to_categorical(label, 50)

train_pct = 0.8
n_files = len(data)
n_train = int(n_files*train_pct)

train = np.random.choice(n_files, n_train, replace=False)
        
# split on training indices
training_idx = np.isin(range(n_files), train)
training_set = np.array(data)[training_idx]
training_label = np.array(label)[training_idx]
validation_set = np.array(data)[~training_idx]
validation_label = np.array(label)[~training_idx] 

n_training_samples = len(training_set) 
n_validation_samples = len(validation_set)

generator = image.ImageDataGenerator(rescale=1./255) 

training_gen = generator.flow(training_set, training_label)
validation_gen = generator.flow(validation_set, validation_label)


base_model = Sequential()
base_model.add(Conv2D(10, kernel_size = (1,1), padding = 'same', activation = 'relu', input_shape = (height, width, 1)))
base_model.add(Conv2D(3, kernel_size = (1,1), padding = 'same', activation = 'relu'))

vgg16_model = VGG16(weights='imagenet', include_top=False )

num_layers_to_freeze = 20 
for layer in vgg16_model.layers[:num_layers_to_freeze]:
    layer.trainable = False

base_model.add(vgg16_model)

top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(50, activation='softmax'))

for layer in top_model.layers:
    layer.trainable = False

model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
model.summary()
model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                      loss='categorical_crossentropy',)

batch_size = 5
epochs = 200


model.fit_generator(
    training_gen,
    steps_per_epoch=n_training_samples/batch_size,
    epochs=epochs,
    validation_data=validation_gen,
    validation_steps=n_validation_samples/batch_size,
    callbacks=[])
