#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 12:37:31 2018
AGE CLASSIFICATION
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
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.applications import InceptionV3
from keras.optimizers import Adam

main_dir = '/home/kdg/kdg_projects/Age/'
folds = []
for i in range(5):
    folds.append(pd.read_csv(main_dir+'fold_'+str(i)+'_data.txt', sep='\t'))
    print(len(folds[i])) # 4484 3730 3894 3446 3816
# merge to total dataframe
data_df = pd.DataFrame()
for i in range(len(folds)):
    df = pd.DataFrame(folds[i])
    data_df = pd.concat([data_df, df])    # (19370, 12)

data_df.age.unique()    
""" array(['(25, 32)', '(38, 43)', '(4, 6)', '(60, 100)', '(15, 20)',
       '(48, 53)', '(8, 12)', '(0, 2)', 'None', '(38, 48)', '35', '3',
       '55', '58', '22', '13', '45', '36', '23', '(38, 42)', '(8, 23)',
       '(27, 32)', '57', '56', '2', '29', '34', '42', '46', '32'], dtype=object) 
Н-ДА...МНОГО МУК ПРИМУ :(  """
Serie = cat_count(data_df, 'age')
""" Чистка мусора. Оставляю только нужное """
data_clean = data_df[data_df.age=='(25, 32)']
for item in ['(38, 43)','(4, 6)', '(60, 100)', '(15, 20)', '(48, 53)',
             '(8, 12)', '(0, 2)']:
    data_clean = pd.concat([data_clean, data_df[data_df.age == item]])
data_clean.to_csv('/home/kdg/kdg_projects/Age/jew.csv', sep=';')
#                                 
                         
"""ПОПРОБУЮ ЛЮБИМЫЙ ВИКИ """
Y = pd.read_csv('/home/kdg/kdg_projects/wiki.csv', sep=';')      
# AGE creating
age = []
for i in range(len(Y)):
    print(i, Y[' photo_taken'][i], Y.dob[i][:4])
    age.append(Y[' photo_taken'][i] - int(Y.dob[i][:4]))
Y['age'] = age
# Y['age'].loc[lambda s: s == 1363]
""" Age categories:
    baby 0-2, child 3-12, teenager 13-19, young 20-35, middle 36-50, 
    senior 51-65, old > 65 """
def cat_count(df, cat): # подсчет в категориях
    category, count = [], []
    for item in df[cat].unique():
        category.append(item)
        count.append(len(df[df[cat] == item]))
    return pd.Series(count, index=category)

age_cat = []
for i in range(len(Y)):
    if Y.age.iloc[i] < 3: age_cat.append('baby')
    elif Y.age.iloc[i] < 13: age_cat.append('child') 
    elif Y.age.iloc[i] < 20: age_cat.append('teenager')  
    elif Y.age.iloc[i] < 26: age_cat.append('young25')
    elif Y.age.iloc[i] < 31: age_cat.append('young30')
    elif Y.age.iloc[i] < 36: age_cat.append('young35')
    elif Y.age.iloc[i] < 51: age_cat.append('middle') 
    elif Y.age.iloc[i] < 66: age_cat.append('senior')
    else: age_cat.append('old') 
Y['age_cat'] = age_cat    
cat_count('age_cat') # 11419 young30, 11975 middle, 7614 senior, 6968 young35
# 4954 old, 14492 young25, 2959 teenager, 216 baby, 234 child
""" Просто по десятилетиям """
age_cat2 = []
for i in range(len(Y)):
    if Y.age.iloc[i] < 11: age_cat2.append('0-10')
    elif Y.age.iloc[i] < 21: age_cat2.append('11-20') 
    elif Y.age.iloc[i] < 31: age_cat2.append('21-30')  
    elif Y.age.iloc[i] < 41: age_cat2.append('31-40')
    elif Y.age.iloc[i] < 51: age_cat2.append('41-50')
    elif Y.age.iloc[i] < 61: age_cat2.append('51-60')
    elif Y.age.iloc[i] < 71: age_cat2.append('61-70') 
    elif Y.age.iloc[i] < 81: age_cat2.append('71-80')
    else: age_cat2.append('80+') 
Y['age_cat2'] = age_cat2    
cat_count('age_cat2') # 24253 21-30, 11522 31-40, 5536 51-60, 7421 41-50,
# 1973 71-80, 3632 61-70, 4697 11-20, 370 0-10, 1427 80+
''' ДАННЫЕ КРАЙНЕ НЕСБАЛАНСИРОВАНЫ!!! '''
#
Y.to_csv(main_dir+'wiki.csv', sep=';')
Y = pd.read_csv('/home/kdg/kdg_projects/Age/wiki.csv', sep=';', index_col=0)
#
# path to source
pic_path = Y[' full_path']
# assembling all files to dst directory
dst = main_dir +'wiki_age/'  
src = main_dir +'wiki_crop/'   
for i, path in enumerate(pic_path):
    shutil.copy2(src+path[1:], dst)  

# Расчет индексов наборов данных для обучения, приверки и тестирования
test_data = 0.15    # Часть набора данных для тестирования
val_data = 0.15     # Часть набора данных для проверки
""" Функция копирования изображений в заданный каталог. 
Изображения разного возраста копируются в отдельные подкаталоги
Подкаталоги создал ВРУЧНУЮ!!! """
def cat_dirs(cat):
    for item in Y[cat].unique():
        start_val_data_idx = int(len(Y[Y[cat] == item]) * (1 - val_data - test_data))
        start_test_data_idx = int(len(Y[Y[cat] == item]) * (1 - test_data))
         
        s = 0
        for i in Y[Y[cat] == item].index:
            p1 = dst + Y[' full_path'].loc[i][4:]   # sourse image
            p2 = dst + cat + '/' + item             # destination dir
            shutil.copy2(p1, p2)
            #print(p1, p2)
            if s < start_val_data_idx:
                shutil.copy2(p1, dst + cat + '/train/' + item)
            elif s < start_test_data_idx:
                shutil.copy2(p1, dst + cat + '/val/' + item)   
            else: shutil.copy2(p1, dst + cat + '/test/' + item)    
            s += 1
#            
cat_dirs('age_cat')
#cat_dirs('age_cat2')    
""" Распознавание возраста на изображениях с помощью предварительно 
обученной нейронной сети VGG16 """
from keras.applications import VGG16
# Размер мини-выборки
batch_size = 64

# Загружаем предварительно обученную нейронную сеть
vgg16_net = VGG16(weights='imagenet', 
                  include_top=False, # НЕ БУДЕТ КЛАССИФИКАЦИИ!
                  input_shape=(150, 150, 3))
# "Замораживаем" веса предварительно обученной нейронной сети VGG16
vgg16_net.trainable = False # Изображения мужчин/женщин уже там есть
vgg16_net.summary()
# Создаем составную нейронную сеть на основе VGG16
model = Sequential()
# Добавляем в модель сеть VGG16 вместо слоев свертки-пулинга;
# сами формируем лишь слои классификации
model.add(vgg16_net)    # выделение признаков
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(9))     # 9 categories
"""model.add(Activation('sigmoid')) # Scalar REGRESSION """
# Компилируем составную нейронную сеть
model.compile(loss='categorical_crossentropy', 
              optimizer=Adam(lr=1e-5), # Низкая скорость обучения
              # если сделать больше - алгоритм не сойдется, т.к. он уже обучен
              # мы просто потеряем уже существующие веса
              metrics=['accuracy'])
""" GENERATORS """
img_width, img_height = 150, 150    # Размеры изображения
datagen = ImageDataGenerator(rescale=1. / 255)
train_dir = dst + 'age_cat/train'
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')   # Found 42577 images belonging to 9 classes.
val_dir = dst + 'age_cat/val'
val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')   # Found 9124 images belonging to 9 classes
test_dir = dst + 'age_cat/test'
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')   # Found 9130 images belonging to 9 classes
"""
Обучаем модель с использованием генераторов """
nb_train_samples = 42577        # Количество изображений для обучения
nb_validation_samples = 9124    # Количество изображений для проверки
nb_test_samples = 9130          # Количество изображений для тестирования
#
startTime = time.time()
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=2, # Поскольку сеть уже обучена
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size)
print("Elapsed time: {:.3f} sec".format(time.time() - startTime))
startTime = time.time()
scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))


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

""" Распознавание возраста на изображениях с помощью предварительно 
обученной нейронной сети VGG19 """
from keras.applications import VGG19
# Загружаем предварительно обученную нейронную сеть
vgg19_net = VGG19(weights='imagenet', 
                  include_top=False, # НЕ БУДЕТ КЛАССИФИКАЦИИ!
                  input_shape=(150, 150, 3))
# "Замораживаем" веса предварительно обученной нейронной сети VGG16
vgg19_net.trainable = False # Изображения мужчин/женщин уже там есть
vgg19_net.summary()
# Создаем составную нейронную сеть на основе VGG16
model19 = Sequential()
# Добавляем в модель сеть VGG16 вместо слоев свертки-пулинга;
# сами формируем лишь слои классификации
model19.add(vgg19_net)    # выделение признаков
model19.add(Flatten())
model19.add(Dense(256))
model19.add(Activation('sigmoid'))
model19.add(Dropout(0.5))
model19.add(Dense(9))     # 9 categories
model19.add(Activation('softmax'))
# Компилируем составную нейронную сеть
model19.compile(loss='categorical_crossentropy', 
              optimizer=Adam(lr=1e-5), # Низкая скорость обучения
              # если сделать больше - алгоритм не сойдется, т.к. он уже обучен
              # мы просто потеряем уже существующие веса
              metrics=['accuracy'])
#
startTime = time.time()
model19.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=5, # Поскольку сеть уже обучена
    validation_data=val_generator, #callbacks = callbacks,
    validation_steps=nb_validation_samples // batch_size)
print("Elapsed time: {:.3f} sec".format(time.time() - startTime))

startTime = time.time()
scores = model19.evaluate_generator(test_generator, nb_test_samples // batch_size)
print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))
print("Elapsed time: {:.3f} sec".format(time.time() - startTime))
