# -*- coding: utf-8 -*-
"""
Encoder(単語→ID)
@author: D. Kishikawa
"""

import csv
import numpy as np

f = open('list.txt','r')
reader = csv.reader(f)
    

word_list = []
id_list = []
dict_list = []
i=0
    
    
for row in reader:
    word_list.append(row)
    id_list.append(i)
    i = i + 1
        
for N in range(len(word_list)):
    string = "".join(word_list[N])
    dict_list.append((string, id_list[N]))
        
dicGlobal = dict(dict_list)
print("dictionary generated")
f.close()
    
def encode(tango):

    #ID
    try:
        tango_id = dicGlobal[tango]
    except:
        tango_id = -1
    
    return tango_id

if __name__ == '__main__':
    data = []

    with open("corpus2.csv","r") as f2:
        reader = csv.reader(f2)
    
        for row in reader:
            data.append(row)
            
    id_list = np.zeros((3, len(data)))
    
    # 1. Process word -> digit
    for i in range(len(data)):
        print("processing {}...".format(i))
        row = data[i]
        id_1 = encode(row[0])
        id_2 = encode(row[1])
        try:
            num = int(row[2])
        except:
            num = 1
        if id_1 == -1:
            id_list[0][i] = int(-1)
        else:
            id_list[0][i] = int(id_1)
        if id_2 == -1:
            id_list[1][i] = int(-1)
        else:
            id_list[1][i] = int(id_2)
        
        try:        
            id_list[2][i] = num
        except ValueError:
            id_list[2][i] = 1
		
    #2. Sum id_list[0][i]
    id_sum = np.zeros(11)
    
    for s in range(len(data)):
        print("calc sum {}...".format(s))
        id_sum[int(id_list[0][s])] = id_sum[int(id_list[0][s])] + id_list[2][s]
        
    #3. calc trans Prob
    for n in range(len(data)):
        print("calc P {}...".format(n))
        try:
            id_list[2][n] = id_list[2][n] / id_sum[int(id_list[0][n])]
        except:
            id_list[2][n] = 0
    #4. write to csv
    np.savetxt('p.csv', id_list.T, delimiter=',')


    
    print(id_list)
