#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 09:57:53 2018
AGE - Regression
@author: kdg
"""
import shutil
import os
import pandas as pd
import numpy as np
import time
""" Timing """
startTime = time.time()
print("Elapsed time: {:.3f} sec".format(time.time() - startTime))

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications import InceptionV3
from keras.optimizers import Adam

main_dir = '/home/kdg/kdg_projects/Age/wiki_age/'

""" ПОПРОБУЮ ЛЮБИМЫЙ ВИКИ """
Y = pd.read_csv('/home/kdg/kdg_projects/Age/wiki.csv', sep=';', index_col=0)
#
# path to source
pic_path = Y[' full_path'] 
# Расчет индексов наборов данных для обучения, приверки и тестирования
test_data = 0.15    # Часть набора данных для тестирования
val_data = 0.15     # Часть набора данных для проверки
# Каталог с данными для обучения
train_dir = '/home/kdg/kdg_projects/Age/wiki_age/train'
# Каталог с данными для проверки
val_dir = '/home/kdg/kdg_projects/Age/wiki_age/val' 
# Каталог с данными для тестирования
test_dir = '/home/kdg/kdg_projects/Age/wiki_age/test'
""" Функция создания каталога с подкаталогами по названию классов: """
def create_directory(dir_name, category):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    for item in category.unique():
        os.makedirs(os.path.join(dir_name, str(item)))    
# Создание структуры каталогов для обучающего, 
# проверочного и тестового набора данных    
create_directory(train_dir, Y.age)
create_directory(val_dir, Y.age)
create_directory(test_dir, Y.age)        
""" Kопированиe изображений в заданный каталог. 
Изображения разного возраста копируются в отдельные подкаталоги """
for item in Y.age.unique():
    start_val_data_idx = int(len(Y[Y.age == item]) * (1 - val_data - test_data))
    start_test_data_idx = int(len(Y[Y.age == item]) * (1 - test_data))
      
    s = 0
    for i in Y[Y.age == item].index:
        p1 = main_dir + Y[' full_path'].loc[i][4:]   # sourse image
        if s < start_val_data_idx:
            shutil.copy2(p1, main_dir + 'train/' + str(item))
        elif s < start_test_data_idx:
            shutil.copy2(p1, main_dir + 'val/' + str(item))   
        else: shutil.copy2(p1, main_dir + 'test/' + str(item))    
        s += 1            
#               
""" Распознавание возраста на изображениях с помощью предварительно 
обученной нейронной сети InceptionV3 """
# Размер мини-выборки
batch_size = 64
# Загружаем предварительно обученную нейронную сеть
Inc_net = InceptionV3(weights='imagenet', 
                  include_top=False, # НЕ БУДЕТ КЛАССИФИКАЦИИ!
                  input_shape=(150, 150, 3))
# "Замораживаем" веса предварительно обученной нейронной сети VGG16
Inc_net.trainable = False # Изображения мужчин/женщин уже там есть
Inc_net.summary()
# Создаем составную нейронную сеть на основе VGG16
model = Sequential()
# Добавляем в модель сеть InceptionV3 вместо слоев свертки-пулинга;
# сами формируем лишь слои классификации
model.add(Inc_net)    # выделение признаков
model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))     # Scalar REGRESSION """
# Компилируем составную нейронную сеть
model.compile(loss='mse', 
              optimizer=Adam(lr=1e-5), # Низкая скорость обучения
              # если сделать больше - алгоритм не сойдется, т.к. он уже обучен
              # мы просто потеряем уже существующие веса
              metrics=['mae'])

"""
ANOTHER WAY - flow_from_dataframe !!!!  """

filename = []
for i in Y.index:
    filename.append(Y[' full_path'][i][4:])
Y['filename'] = filename

""" Вспомогательные датафреймы """
start_val_data_idx = int(len(Y) * (1 - val_data - test_data))
start_test_data_idx = int(len(Y) * (1 - test_data))
#
train_df = Y[:start_val_data_idx]
val_df = Y[start_val_data_idx : start_test_data_idx].reset_index()
test_df = Y[start_test_data_idx:].reset_index()

""" Kопированиe изображений в заданный каталог """
s = 0
for i in Y.index:
    p1 = main_dir + Y.filename.loc[i]  # sourse image
    if s < start_val_data_idx:
        shutil.copy2(p1, main_dir + 'TRAIN/')
    elif s < start_test_data_idx:
        shutil.copy2(p1, main_dir + 'VAL/')   
    else: shutil.copy2(p1, main_dir + 'TEST/')    
    s += 1            
#               
""" GENERATORS """
train_dir = main_dir + 'TRAIN'
train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col='filename', y_col='age',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='sparse')   # Found 42501 images belonging to 9 classes.
val_dir = main_dir + 'VAL'
val_generator = datagen.flow_from_dataframe(val_df,
    val_dir, x_col='filename', y_col='age',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='sparse')   # Found 9125 images belonging to 9 classes
test_dir = main_dir + 'TEST'
test_generator = datagen.flow_from_dataframe(test_df,
    test_dir, x_col='filename', y_col='age',
    target_size=(img_width, img_height),
    batch_size=batch_size, 
    class_mode='sparse')   # Found 9125 images belonging to 9 classes

nb_train_samples = 42501        # Количество изображений для обучения
nb_validation_samples = 9125    # Количество изображений для проверки
nb_test_samples = 9125          # Количество изображений для тестирования
#
startTime = time.time()
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=10, # Поскольку сеть уже обучена
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size)
print("Elapsed time: {:.3f} sec".format(time.time() - startTime))


startTime = time.time()
scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print("Hа тестовых данных: %.2f%%" % (scores[1]))
"""
 Тонкая настройка сети (fine tuning). Все предварительные данные взяты
 из age_clasfn.py
"Размораживаем" последний сверточный блок сети inception_v3 """
Inc_net.trainable = True
trainable = False
for layer in Inc_net.layers[-31:]:
    layer.trainable = True

model.compile(loss='mse', 
              optimizer=Adam(lr=1e-5),
              metrics=['mae'])


"""Testing by real image
from keras.preprocessing.image import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
# load image
img = image.load_img('/home/kdg/kdg_projects/Age/IMG_20180911_145533__01.jpg',
                     target_size = (img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
x = preprocess_input(x)
pred = model.predict(x)     # recognizing object
#[0.00166305, 0.00300411, 0.2631992 , 0.0865822 , 0.22264355,
#        0.02975379, 0.11613011, 0.07298661, 0.10546014]
Y.age_cat.unique() # ['young30', 'middle', 'senior', 'young35', 'old', 
                   # 'young25', 'teenager', 'baby', 'child']
""" Вроде как сеньор=26%? больше всего... СООТВЕТСТВУЕТ)) 
IMG_20181005_102611__01.jpg """
img = image.load_img('/home/kdg/kdg_projects/Age/IMG_20180725_090123__01.jpg',
                     target_size = (img_width, img_height))    
# [0.00331942, 0.00882073, 0.0945544 , 0.00734602, 0.02299088,
#        0.09575962, 0.5137222 , 0.31918275, 0.20655406]
# c вероятностью 51% - тинейджер?
"""
"""Testing by real image FACES 
Идея в следующем: взять все фото отдельной личности из набора Faces,
и усреднить полученные предсказания о возрасте. 
Возьму по 1 лицу из каждой категории: """
face_dir = '/home/kdg/kdg_projects/Age/faces/'
create_directory(face_dir, data_clean.age)
pred = []
for item in os.listdir(path=face_dir+'(25, 32)'):
    photo = image.load_img(face_dir+'(25, 32)/'+item, 
            target_size = (img_width, img_height))  
    x = image.img_to_array(photo)
    x = np.expand_dims(x, axis = 0)
    # x /= 255.
    x = preprocess_input(x)
    pred.append(model.predict(x))
np.mean(pred)
#data_clean[data_clean.age=='(0, 2)'].original_image # 3752    10356952985_e2d68b3afc_o.jpg

"""
# Генерируем описание модели в json
model_json = model.to_json()
json_file = open('IncV3_regr_model.json', 'w')
json_file.write(model_json)
json_file.close()
# Weights saving
model.save_weights('IncV3_regr_model.h5')    
#
json_file = open('IncV3_regr_model.json', 'r')
model_json = json_file.read()
json_file.close()
#
from keras.models import model_from_json
loaded_model = model_from_json(model_json)
loaded_model.load_weights('IncV3_regr_model.h5')
"""

