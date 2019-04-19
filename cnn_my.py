# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 22:28:31 2019

@author: tdpco
"""


# Importing the Libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image

# Initializing the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3,
                             input_shape=(64, 64, 3),
                             activation='relu'))
# Step 2 - Max Pooling
classifier.add(MaxPool2D(pool_size=(2, 2)))

# Step 3 - Flatten
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(output_dim=1,
                     activation='sigmoid'))

# Step 5 - Compiling the CNN
classifier.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

# Part 2 - Fitting the CNN
training_datagen = ImageDataGenerator(rescale=1./255,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = training_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = training_datagen.flow_from_directory('dataset/test_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

classifier.fit_generator(training_set,
                         samples_per_epoch=8000,
                         nb_epoch=25,
                         validation_data=test_set,
                         nb_val_samples=2000)

# Save the model
classifier.save('model_cnn_72.h5')

# Predicting a unseen image
test_image = image.load_img('single_prediction/cat_or_dog_1.jpg',
                            target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
results = classifier.predict(test_image)

if (results == 0):
    print("\n\nCat")
else:
    print("\n\nDogs")

# training_set.class_indices
