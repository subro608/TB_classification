import os,glob
import numpy as np
import cv2
import random
import glob
import pickle
import tensorflow as tf
import argparse
import re
import datetime

def files_list(location):
    _dir = location
    dir1 = os.path.join(_dir,"*.png") 
    _files = glob.glob(dir1)
    return _files

def create_dic(normal_files,tb_files):
    file_dic = {}
    for i in range(len(normal_files)):
        file_dic[normal_files[i]] = [0,1]
    for i in range(len(tb_files)):
        file_dic[tb_files[i]] = [1,0]
    l_dic = list(file_dic.items())
    random.Random(4).shuffle(l_dic)
    return l_dic

def create_data(l_file):
    _data = []
    _labels = []

    for i in range(len(l_file)):
        file_name,label = l_file[i]
        img = cv2.imread(file_name)
        try:
            img = cv2.resize(img,(224,224),interpolation = cv2.INTER_CUBIC)
            img = img.astype('float32')/255.0
            _data.append(img)
            _labels.append(label)

        except:
            print(i,file_name)
            print("Not possible")        
    new_data = np.array(_data)
    new_labels = np.array(_labels)
    return new_data, new_labels 



# if __name__ == "__main__":
#     normal_files = files_list("/content/TB_Chest_Radiography_Database/Normal")
#     tb_files = files_list("/content/TB_Chest_Radiography_Database/Tuberculosis")
#     print(len(normal_files),len(tb_files))
#     normal_files.sort()
#     tb_files.sort()
#     train_dic = create_dic(normal_files[:int(0.6 * len(normal_files))],tb_files[:int(0.6 * len(tb_files))])
#     val_dic = create_dic(normal_files[int(0.6 * len(normal_files)):int(0.7 * len(normal_files))],tb_files[int(0.6 * len(tb_files)):int(0.7 * len(tb_files))])
#     test_dic = create_dic(normal_files[int(0.7 * len(normal_files)):len(normal_files)],tb_files[int(0.7 * len(tb_files)):len(tb_files)])
#     train_data, train_labels = create_data(train_dic)
#     val_data, val_labels = create_data(val_dic)
#     test_data, test_labels = create_data(test_dic)
#     print(train_data.shape,val_data.shape,test_data.shape)