#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 10:13:33 2018
ОБЪЕДИНЕНИЕ Flickr & Wiki
@author: kdg
"""
import shutil
import os
import pandas as pd
import numpy as np
import time
# Wiki
Y = pd.read_csv('/home/kdg/kdg_projects/Age/wiki.csv', sep=';', index_col=0)  
# Flickr
data = pd.read_csv('/home/kdg/kdg_projects/Age/jew.csv', sep=';', index_col=0)
data.age.unique() # Flickr categories '(25, 32)', '(38, 43)', '(4, 6)',
                  # '(60, 100)', '(15, 20)','(48, 53)', '(8, 12)', '(0, 2)'
""" Wiki to Flickr """
flkr_cat, flkr_ind = [], []
for i in range(len(Y)):
    if Y.age.iloc[i] < 3: 
        flkr_cat.append('(0, 2)')
        flkr_ind.append(i)
    elif Y.age.iloc[i] < 7: 
        flkr_cat.append('(4, 6)') 
        flkr_ind.append(i)
    elif Y.age.iloc[i] < 13: 
        flkr_cat.append('(8, 12)')  
        flkr_ind.append(i)
    elif (Y.age.iloc[i] > 14) & (Y.age.iloc[i] < 21): 
        flkr_cat.append('(15, 20)')
        flkr_ind.append(i)
    elif (Y.age.iloc[i] > 24) & (Y.age.iloc[i] < 33): 
        flkr_cat.append('(25, 32)')
        flkr_ind.append(i)
    elif (Y.age.iloc[i] > 37) & (Y.age.iloc[i] < 44): 
        flkr_cat.append('(38, 43)')
        flkr_ind.append(i)
    elif (Y.age.iloc[i] > 47) & (Y.age.iloc[i] < 54): 
        flkr_cat.append('(48, 53)') 
        flkr_ind.append(i)
    elif Y.age.iloc[i] > 59: 
        flkr_cat.append('(60, 100)')
        flkr_ind.append(i)
    else: pass
Y1 = Y.iloc[flkr_ind]   # extracting data
Y1['flkr_cat'] = flkr_cat   
cat_count(data, 'age') + cat_count(Y1, 'flkr_cat') # function from age_clasfn

""" Функция создания каталога с подкаталогами по названию классов: """
def create_directory(dir_name, category):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    for item in category.unique():
        os.makedirs(os.path.join(dir_name, str(item)))  

main_dir = '/home/kdg/kdg_projects/Age/WikiFlickr/'
create_directory(main_dir, Y1.flkr_cat)

""" Kопированиe изображений в заданный каталог. 
Изображения разного возраста копируются в отдельные подкаталоги """
source = '/home/kdg/kdg_projects/Age/wiki_age/'
for item in Y1.flkr_cat.unique():
    for i in Y1[Y1.flkr_cat == item].index:
        p1 = source + Y1[' full_path'].loc[i][4:]   # sourse image
        shutil.copy2(p1, main_dir + item)
source2 = '/home/kdg/kdg_projects/Age/faces/'        
for item in data.age.unique():
    for i in range(len(data[data.age == item])):
        p1 = source2 + data.user_id.iloc[i] + '/' + \
        'coarse_tilt_aligned_face.' + str(data.face_id.iloc[i]) + \
        '.' + data.original_image.iloc[i]   
        shutil.copy2(p1, main_dir + item)               

""" Union Data """
union_len = len(Y1) + len(data) # 56045
Union = pd.DataFrame(index=range(union_len))
Union['age'] = np.hstack((Y1.flkr_cat.values, data.age.values))
Union['img'] = np.hstack((Y1[' full_path'].values, data.original_image.values))
