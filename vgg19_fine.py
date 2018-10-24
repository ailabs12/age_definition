#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 08:41:14 2018

@author: kdg
 Тонкая настройка сети (fine tuning). Все предварительные данные взяты
 из age_clasfn.py
"Размораживаем" последний сверточный блок сети VGG19 """
vgg19_net.trainable = True
trainable = False
for layer in vgg19_net.layers:
    if layer.name == 'block5_conv1':
        trainable = True
    layer.trainable = trainable
model19.summary()    

model19.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=1e-5), 
              metrics=['accuracy'])

startTime = time.time()
model19.fit_generator(train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=15,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size)    
print("Elapsed time: {:.3f} sec".format(time.time() - startTime))
# Аккуратность на тестовых данных: ???? %
startTime = time.time()
scores = model19.evaluate_generator(test_generator, nb_test_samples // batch_size)
print("Аккуратность на тестовых данных: %.2f%%" % (scores[1]*100))
print("Elapsed time: {:.3f} sec".format(time.time() - startTime))

""" АНАЛИЗ ПРИЗНАКОВ, ИЗВЛЕЧЕННЫХ НЕЙРОСЕТЬЮ """
train_generator = datagen.flow_from_directory(train_dir,
    target_size=(224, 224), # standart for VGG16
    batch_size=10,
    class_mode=None,    # извлекаем только изображения, без ответов
    shuffle=False)      # тот же порядок, что на диске, без перемешивания
# CNN loading
vgg19 = VGG19(weights='imagenet', 
              include_top=False, # НЕ БУДЕТ КЛАССИФИКАЦИИ!
              input_shape=(224, 224, 3))
# извлечение признаков
features_train = vgg19.predict_generator(train_generator)
features_train.shape 
""" (17670, 7, 7, 512)
формат выходного значения сверточной части VGG:
17670 картинок, 512 карт признаков размером 7х7 """
# saving features    
np.save(open('features_train19.npy','wb'), features_train)    
#
val_generator = datagen.flow_from_directory(val_dir,
    target_size=(224, 224), # standart for VGG19
    batch_size=10,
    class_mode=None,    # извлекаем только изображения, без ответов
    shuffle=False) 
startTime = time.time()
features_val = vgg19.predict_generator(val_generator)
features_val.shape  # (3788, 7, 7, 512)
print("Elapsed time: {:.3f} sec".format(time.time() - startTime))
# Elapsed time: 150.428 sec
np.save(open('features_val19.npy','wb'), features_val)  
#
test_generator = datagen.flow_from_directory(test_dir,
    target_size=(224, 224), # standart for VGG19
    batch_size=10,
    class_mode=None,    # извлекаем только изображения, без ответов
    shuffle=False) 
features_test = vgg19.predict_generator(test_generator)
features_test.shape  # (3788, 7, 7, 512)
np.save(open('features_test19.npy','wb'), features_test)  

# Генерируем правильные ответы, 0 и 1 - метки классов
labels_train = np.array([0]*(nb_train_samples //2) + [1]*(nb_train_samples //2))
labels_val = np.array([0]*(nb_validation_samples //2) + [1]*(nb_validation_samples //2))
labels_test = np.array([0]*(nb_test_samples //2) + [1]*(nb_test_samples //2))
# Создаем простую сеть для классфикации
vgg19plus = Sequential()
vgg19plus.add(Flatten(input_shape=features_train.shape[1:]))
vgg19plus.add(Dense(256, activation ='sigmoid'))
vgg19plus.add(Dropout(0.5))
vgg19plus.add(Dense(1, activation ='softmax'))
#Компилируем нейронную сеть
vgg19plus.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# RUN model
startTime = time.time()
vgg19plus.fit(features_train, labels_train, batch_size=64, 
                epochs=15, callbacks = callbacks,
                validation_data=(features_val, labels_val), verbose=2)  
print("Elapsed time: {:.3f} sec".format(time.time() - startTime))

""" TensorBoard visualization """
from keras.callbacks import TensorBoard
# callback creating
callbacks = [TensorBoard(log_dir = main_dir+'tb_logs', histogram_freq = 1, 
                         write_images = True)]

