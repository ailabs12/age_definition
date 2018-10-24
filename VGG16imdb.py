#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 12:05:14 2018
Распознавание возраста на изображениях с помощью предварительно 
обученной нейронной сети VGG16 
Используется набор imdb, причем из него взята только категория «взрослые» 
(20-44). Она разбита, в свою очередь, на 12 подкатегорий по 2 года — в целях
эксперимента, НА СКОЛЬКО точно может работать ИНС.
При этом имеющийся ранее набор wiki преобразован аналогично, и будет 
использоваться в качестве тестового — поскольку имеет другую «физическую 
природу». Также есть мысль использовать для теста как сбалансированный 
по категориям сет, так и полный (несбалансированный, «как есть»).
@author: kdg
"""
import time
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.applications import VGG16
# Размер мини-выборки
batch_size = 32
# Загружаем предварительно обученную нейронную сеть
vgg16_net = VGG16(weights='imagenet', 
                  include_top=False,
                  input_shape=(224, 224, 3)) # НЕ БУДЕТ КЛАССИФИКАЦИИ!                 
# "Замораживаем" веса предварительно обученной нейронной сети VGG16
vgg16_net.trainable = False # Изображения мужчин/женщин уже там есть
vgg16_net.summary()
# Создаем составную нейронную сеть на основе VGG16
model = Sequential()
# Добавляем в модель сеть VGG16 вместо слоев свертки-пулинга;
# сами формируем лишь слои классификации
model.add(vgg16_net)    # выделение признаков
model.add(Flatten())
model.add(Dense(512))   # поскольку категорий больше, решил добавить нейронов
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(12))     # 12 categories
#model.add(Activation('softmax')) # ЗАДАЧА СКАЛЯРНОЙ РЕГРЕССИИ!!!
# Компилируем составную нейронную сеть
model.compile(loss='categorical_crossentropy', 
              optimizer=Adam(), # Низкая скорость обучения
              # если сделать больше - алгоритм не сойдется, т.к. он уже обучен
              # мы просто потеряем уже существующие веса
              metrics=['accuracy'])
""" GENERATORS """
img_width, img_height = 224, 224    # Размеры изображения
datagen = ImageDataGenerator(rescale=1. / 255)
main_dir = '/home/kdg/kdg_projects/Age'

train_dir = main_dir +'/imdb_age/ADULT/train/'
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')   # Found 208800 images belonging to 12 classes.

val_dir = main_dir +'/imdb_age/ADULT/val/'
val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')   # Found 31200 images belonging to 12 classes

test_dir = main_dir +'/imdb_age/ADULT/test1/' # Found 40526 images belonging to 12 classes
#test_dir = './imdb_age/ADULT/test2/'
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')   
"""
Обучаем модель с использованием генераторов """
nb_train_samples = 208800       # Количество изображений для обучения
nb_validation_samples = 31200   # Количество изображений для проверки
nb_test_samples = 40526         # Количество изображений для тестирования
#
startTime = time.time()
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=1, # Поскольку ВСЕ ОЧЕНЬ ДОЛГО... :()
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size)
print("Elapsed time: {:.3f} sec".format(time.time() - startTime))
#
""" FEATURES EXTRACTING """
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,    # только сами данные, без инфы о классе
    shuffle=False)      # извлекаются в порядке хранения на диске
startTime = time.time()
features_train = vgg16_net.predict_generator(train_generator,
                    nb_train_samples // batch_size)
print("Elapsed time: {:.3f} sec".format(time.time() - startTime))
features_train.shape 

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')  
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')   

"""
startTime = time.time()
scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))
print("Elapsed time: {:.3f} sec".format(time.time() - startTime))
"""
Testing by real image """
from keras.preprocessing.image import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
# load image
img = image.load_img(main_dir+'IMG_20180911_145533__01.jpg',
                     target_size = (img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
# x /= 255.  ???
x = preprocess_input(x)
pred = model.predict(x)     # recognizing object
#[0.00166305, 0.00300411, 0.2631992 , 0.0865822 , 0.22264355,
#        0.02975379, 0.11613011, 0.07298661, 0.10546014]
Y.age_cat.unique() # ['young30', 'middle', 'senior', 'young35', 'old', 
                   # 'young25', 'teenager', 'baby', 'child']
""" Вроде как сеньор=26%? больше всего... СООТВЕТСТВУЕТ)) 
IMG_20181005_102611__01.jpg """
img = image.load_img(main_dir+'IMG_20180725_090123__01.jpg',
                     target_size = (img_width, img_height))    
# [0.00331942, 0.00882073, 0.0945544 , 0.00734602, 0.02299088,
#        0.09575962, 0.5137222 , 0.31918275, 0.20655406]
# c вероятностью 51% - тинейджер?


# Генерируем описание модели в json
model_json = model.to_json()
json_file = open('age_cat_model.json', 'w')
json_file.write(model_json)
json_file.close()
# Weights saving
model.save_weights('age_cat_model.h5')    
