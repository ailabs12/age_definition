#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 15:46:22 2018
DATASET IMDB + WIKI
Xception
Dataframe 'imdb' is created in imdb_to_csv.py module
@author: kdg
"""
from keras.applications.xception import Xception
import shutil
import os
import pandas as pd
import numpy as np
import time

imdb = pd.read_csv("./imdb.csv", sep=';', index_col=0)
""" Age categories (FROM Sedykh. Russian passport rulez):
    child 0-13, teenager 14-19, adult 20-44, senior 45+ """
age_cat = []
for i in range(len(imdb)):
    if imdb.age.iloc[i] < 14: age_cat.append('child') 
    elif imdb.age.iloc[i] < 20: age_cat.append('teenager')  
    elif imdb.age.iloc[i] < 45: age_cat.append('adult') 
    else: age_cat.append('senior') 
imdb['age_cat'] = age_cat   

def cat_count(df, cat): # подсчет в категориях
    Ind, Old = [], []
    for item in df[cat].unique():
        Old.append(len(df[df[cat] == item]))
        Ind.append(item)
    return pd.Series(Old, index=Ind).sort_index()   
 
dstr_cat = cat_count(imdb, 'age_cat') # 19452 teenager, 318291 adult, 8736 child, 113602 senior
distrib = cat_count(imdb, 'age') # age distribution

""" Начну с варианта 4 вышеописанных категорий. 
Данные в категориях приму равными самой маленькой - child, т.е. по 8736 
Кроме того, можно слить с набором wiki - что увеличит данные"""
wiki = pd.read_csv("./wiki.csv", sep=';', index_col=0)
wk = wiki.drop(['dob', ' photo_taken', ' gender', ' name', ' face_location',
        ' face_score', ' second_face_score', 'age_cat', 'age_cat2'], axis=1)
wk_dstr_cat = cat_count(wk, 'age_cat')    
# adult 40526, child 531, senior 16896, teenager 2878
"""
ПЕРЕДУМАЛ... очень уж мал child - прям по названию))
Попробую точно поделить adult - причем wiki-adult м.б. ТЕСТОМ для imdb-adult,
т.к. составляет около 13% """
# Moving ADULT to self directory
for i in imdb[imdb.age_cat == 'adult'].index:
    p1 = './imdb_age/' + imdb.filename.loc[i]  # sourse image
    shutil.copy2(p1, './imdb_age/ADULT/')
 # Расчет индексов наборов данных для обучения, приверки и тестирования
test_data = 0.13    # Часть набора данных для тестирования - Wiki-adult
val_data = 0.13     # Часть набора данных для проверки
""" Функция копирования изображений в заданный каталог. 
Изображения разного возраста копируются в отдельные подкаталоги.
Подкаталоги создал ВРУЧНУЮ!!!
Также создал спец. датафрейм для Adult и разбивку внутри"""
Adult = imdb[imdb['age_cat'] == 'adult']
distribA = cat_count(Adult, 'age')

adult_cat = []
for i in range(len(Adult)):
    if Adult.age.iloc[i] < 23: adult_cat.append('20-22') 
    elif Adult.age.iloc[i] < 25: adult_cat.append('23-24')  
    elif Adult.age.iloc[i] < 27: adult_cat.append('25-26')
    elif Adult.age.iloc[i] < 29: adult_cat.append('27-28')  
    elif Adult.age.iloc[i] < 31: adult_cat.append('29-30')
    elif Adult.age.iloc[i] < 33: adult_cat.append('31-32')  
    elif Adult.age.iloc[i] < 35: adult_cat.append('33-34')
    elif Adult.age.iloc[i] < 37: adult_cat.append('35-36')  
    elif Adult.age.iloc[i] < 39: adult_cat.append('37-38')
    elif Adult.age.iloc[i] < 41: adult_cat.append('39-40')  
    elif Adult.age.iloc[i] < 43: adult_cat.append('41-42')
    else: adult_cat.append('43-44') 
Adult['age_cat'] = adult_cat   
distrA = cat_count(Adult, 'age_cat')
"""
Cutting categories to 20000 size """
excess = []
for cat in Adult.age_cat.unique():
    for i, ind in enumerate(Adult.index[Adult.age_cat == cat]):
        if i >= 20000: excess.append(ind)
Adult_crop = Adult.drop(excess)     # 240000 photos
distrA = cat_count(Adult_crop, 'age_cat')    
"""
To directories... manually created """
def cat_dirs(df, cat):
    for item in df[cat].unique():
        start_val_data_idx = int(len(df[df[cat] == item]) * (1 - val_data))
          
        s = 0
        for i in df[df[cat] == item].index:
            p1 = './imdb_age/' + df.filename.loc[i]   # sourse image
            if s < start_val_data_idx:
                shutil.copy2(p1, './imdb_age/ADULT/train/' + item)
            else: shutil.copy2(p1, './imdb_age/ADULT/val/' + item)    
            s += 1
#            
cat_dirs(Adult_crop, 'age_cat')

""" XCEPTION !!! """
from keras.applications.xception import Xception
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
# Размер мини-выборки
batch_size = 64
# Загружаем предварительно обученную нейронную сеть
Xc_net = Xception(weights='imagenet', 
                  include_top=False, # НЕ БУДЕТ КЛАССИФИКАЦИИ!
                  input_shape=(224, 224, 3))
# "Замораживаем" веса предварительно обученной нейронной сети VGG16
Xc_net.trainable = False # Изображения мужчин/женщин уже там есть
Xc_net.summary()
# Создаем составную нейронную сеть на основе Xception
Xmodel = Sequential()
# Добавляем в модель сеть VGG16 вместо слоев свертки-пулинга;
# сами формируем лишь слои классификации
Xmodel.add(Xc_net)    # выделение признаков
Xmodel.add(Flatten())
Xmodel.add(Dense(1024)) # А! Где наша не пропадала!!!
Xmodel.add(Activation('sigmoid'))
Xmodel.add(Dropout(0.5))
Xmodel.add(Dense(12))     # 12 categories
Xmodel.add(Activation('softmax'))
# Компилируем составную нейронную сеть
Xmodel.compile(loss='categorical_crossentropy', 
              optimizer=Adam(lr=1e-5), # Низкая скорость обучения
              # если сделать больше - алгоритм не сойдется, т.к. он уже обучен
              # мы просто потеряем уже существующие веса
              metrics=['accuracy'])
""" GENERATORS """
img_width, img_height = 224, 224    # Размеры изображения
datagen = ImageDataGenerator(rescale=1. / 255)
train_dir = './imdb_age/ADULT/train/'
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')   # Found 208800 images belonging to 12 classes.
val_dir = './imdb_age/ADULT/val/'
val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')   # Found 31200 images belonging to 12 classes.
test_dir = './imdb_age/ADULT/test/'
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')   # 
"""
Обучаем модель с использованием генераторов """
nb_train_samples = 208800       # Количество изображений для обучения
nb_validation_samples = 31200   # Количество изображений для проверки
nb_test_samples =         # Количество изображений для тестирования
#
startTime = time.time()
Xmodel.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=1, # Поскольку сеть уже обучена
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size)
print("Elapsed time: {:.3f} sec".format(time.time() - startTime))
""" TEST """
startTime = time.time()
scores = model.evaluate_generator(test_generator, nb_test_samples // batch_size)
print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))
print("Elapsed time: {:.3f} sec".format(time.time() - startTime))
