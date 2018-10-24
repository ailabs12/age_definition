#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:47:24 2018

@author: kdg
"""

from keras.applications.mobilenet import MobileNet
mob_net = MobileNet(weights='imagenet', 
                  include_top=False, # НЕ БУДЕТ КЛАССИФИКАЦИИ!
                  input_shape=(128, 128, 3))
mob_net.summary()
batch_size = 16

model = Sequential()
model.add(mob_net)    # выделение признаков
model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(11))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-5), 
              metrics=['accuracy'])
""" GENERATORS """
img_width, img_height = 128, 128    # Размеры изображения
datagen = ImageDataGenerator(rescale=1. / 255)
train_dir = '/home/kdg/kdg_projects/Age/imdb_age/ADULT/train'
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')   # Found 191400 images belonging to 11 classes.
val_dir = '/home/kdg/kdg_projects/Age/imdb_age/ADULT/val'
val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')   # Found 28600 images belonging to 11 classes.
test_dir = '/home/kdg/kdg_projects/Age/imdb_age/ADULT/test1'
#test_dir = '/home/kdg/kdg_projects/Age/imdb_age/ADULT/test2'
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')  # Found 16500 images belonging to 11 classes.
nb_train_samples = 191400        # Количество изображений для обучения
nb_validation_samples = 28600    # Количество изображений для проверки
nb_test_samples = 16500

startTime = time.time()
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=5, # Поскольку сеть уже обучена
    validation_data=val_generator, #callbacks = callbacks,
    validation_steps=nb_validation_samples // batch_size)
print("Elapsed time: {:.3f} sec".format(time.time() - startTime))

""" REALITY SHOW """
from keras.preprocessing.image import image
from keras.applications.inception_v3 import preprocess_input
# load image
photos = ['/home/kdg/kdg_projects/Age/IMG_20180911_145533__01.jpg',
          '/home/kdg/kdg_projects/Age/IMG_20180725_090123__01.jpg',
          '/home/kdg/kdg_projects/Age/IMG_20181005_102611__01.jpg',
          '/home/kdg/kdg_projects/Age/IMG_20181008_094702__01.jpg',
          '/home/kdg/kdg_projects/Age/Screenshot_20181018-100743__01.jpg']
for item in photos:
    img = image.load_img(item, target_size = (img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)
    print(model.predict(x))  
