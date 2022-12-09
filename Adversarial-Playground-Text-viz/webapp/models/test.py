import os

import pickle
for i in range(0,8):
    if(i == 4): continue
    info = pickle.load(open(os.getcwd() + '\\dict\\' + str(i) + '.info', 'rb'))
    dict_word = info['word_index']
    for i,j in dict_word.items():
        if(j == 2):print(i)

model =