#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 09:29:51 2018
Wiki is TEST set
@author: kdg
"""

wiki = pd.read_csv("./wiki.csv", sep=';', index_col=0)
wk = wiki.drop(['dob', ' photo_taken', ' gender', ' name', ' face_location',
        ' face_score', ' second_face_score', 'age_cat', 'age_cat2'], axis=1)
wk_adult = wk[(wk.age > 19) & (wk.age < 45)]   
adult_cat = []
for i in range(len(wk_adult)):
    if wk_adult.age.iloc[i] < 23: adult_cat.append('20-22') 
    elif wk_adult.age.iloc[i] < 25: adult_cat.append('23-24')  
    elif wk_adult.age.iloc[i] < 27: adult_cat.append('25-26')
    elif wk_adult.age.iloc[i] < 29: adult_cat.append('27-28')  
    elif wk_adult.age.iloc[i] < 31: adult_cat.append('29-30')
    elif wk_adult.age.iloc[i] < 33: adult_cat.append('31-32')  
    elif wk_adult.age.iloc[i] < 35: adult_cat.append('33-34')
    elif wk_adult.age.iloc[i] < 37: adult_cat.append('35-36')  
    elif wk_adult.age.iloc[i] < 39: adult_cat.append('37-38')
    elif wk_adult.age.iloc[i] < 41: adult_cat.append('39-40')  
    elif wk_adult.age.iloc[i] < 43: adult_cat.append('41-42')
    else: adult_cat.append('43-44') 
wk_adult['age_cat'] = adult_cat   

distr = cat_count(wk_adult, 'age_cat')
distr.plot(kind='bar')
"""
20-22    6227
23-24    5622
25-26    5334
27-28    4733
29-30    3995
31-32    3287
33-34    2572
35-36    2187
37-38    1819
39-40    1657
41-42    1558
43-44    1535 

To directories... manually created """
for item in wk_adult['age_cat'].unique():
    for i in wk_adult[wk_adult['age_cat'] == item].index:
        p1 = './wiki_age/' + wk_adult[' full_path'].loc[i][4:]   # sourse image
        shutil.copy2(p1, './imdb_age/ADULT/test1/' + item)            
"""
Cutting categories to 1500 size """
excess = []
for cat in wk_adult.age_cat.unique():
    for i, ind in enumerate(wk_adult.index[wk_adult.age_cat == cat]):
        if i >= 1500: excess.append(ind)
wk_adult_crop = wk_adult.drop(excess)     # 18000 photos
"""
To directories... manually created """
for item in wk_adult_crop['age_cat'].unique():
    for i in wk_adult_crop[wk_adult_crop['age_cat'] == item].index:
        p1 = './wiki_age/' + wk_adult_crop[' full_path'].loc[i][4:]   # sourse image
        shutil.copy2(p1, './imdb_age/ADULT/test2/' + item)