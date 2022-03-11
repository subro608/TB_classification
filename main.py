from data import files_list, create_dic, create_data
from model import vgg, densenet, resnet
from fuzzy import sugeno, choquet
from mealpy.swarm_based.GWO import BaseGWO
from sklearn.metrics import accuracy_score
import os,glob
import numpy as np
import cv2
import glob
import pickle
import tensorflow as tf
import argparse
import re
import datetime
from tensorflow.keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract,LeakyReLU,Add,Average,Lambda,MaxPool2D,Dropout,UpSampling2D,Concatenate,Multiply,GlobalAveragePooling2D,Dense,ZeroPadding2D,AveragePooling2D
from tensorflow.keras.layers import concatenate,Flatten,ConvLSTM2D,LayerNormalization,GlobalAveragePooling2D,DepthwiseConv2D,SeparableConv2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler 
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
import tensorflow.keras.backend as K
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog,local_binary_pattern
from skimage import data, exposure
from tensorflow.keras.layers import Layer

def optimize(obj):
    obj_func = obj
    lb = [0]
    ub = [1]
    problem_size = 3
    batch_size = 25
    verbose = True
    epoch = 5
    pop_size = 100
    pc = 0.95
    pm = 0.025

    md1 = BaseGWO(obj_func, lb, ub, problem_size, batch_size, verbose, epoch, pop_size)
    best_pos1, best_fit1, list_loss1 = md1.train()
    print(md1.solution[0])
    print(md1.solution[1])
    return md1.solution[1]

def cho_objective(solution = None):
    print(solution)
    acc = choquet(solution, pred1,pred2,pred3, val_labels)
    return acc

def sug_objective(solution = None):
    acc = sugeno(solution, pred1,pred2,pred3, val_labels)
    return acc


if __name__ == "__main__":
    # Collect the filesnames for both the normal and tuberculosis affected patients 
    normal_files = files_list("/content/TB_Chest_Radiography_Database/Normal")
    tb_files = files_list("/content/TB_Chest_Radiography_Database/Tuberculosis")
    print(len(normal_files),len(tb_files))
    normal_files.sort()
    tb_files.sort()
    # Split the files into train, test and validation set
    train_dic = create_dic(normal_files[:int(0.6 * len(normal_files))],tb_files[:int(0.6 * len(tb_files))])
    val_dic = create_dic(normal_files[int(0.6 * len(normal_files)):int(0.7 * len(normal_files))],tb_files[int(0.6 * len(tb_files)):int(0.7 * len(tb_files))])
    test_dic = create_dic(normal_files[int(0.7 * len(normal_files)):len(normal_files)],tb_files[int(0.7 * len(tb_files)):len(tb_files)])
    train_data, train_labels = create_data(train_dic)
    val_data, val_labels = create_data(val_dic)
    test_data, test_labels = create_data(test_dic)
    print(train_data.shape,val_data.shape,test_data.shape)
    #Train the models on training set
    vgg_model, his_vgg = vgg((224,224,3),2,train_data, train_labels, val_data, val_labels)
    densenet_model, his_dense = densenet((224,224,3),2,train_data, train_labels, val_data, val_labels)
    resnet_model, his_resnet = resnet((224,224,3),2,train_data, train_labels, val_data, val_labels)
    #Make predictions on the validation set
    pred1 = vgg_model.predict(val_data)
    pred2 = densenet_model.predict(val_data)
    pred3 = resnet_model.predict(val_data)
    # Use optimization to get the best set of fuzzy measure values
    sol1 = optimize(cho_objective)
    sol2 = optimize(sug_objective)
    #Make predictions on the Test set
    test_pred1 = vgg_model.predict(test_data)
    test_pred2 = densenet_model.predict(test_data)
    test_pred3 = resnet_model.predict(test_data)
    # Get the final predictions using the fuzzy measure values obtained using the optimization algorithms
    neg_acc_cho = choquet([sol1[0],sol1[1],sol1[2]],test_pred1, test_pred2, test_pred3, test_labels)
    neg_acc_sug = sugeno([sol2[0],sol2[1],sol2[2]],test_pred1, test_pred2, test_pred3, test_labels)
    cho_acc = -1 * neg_acc_cho
    sug_acc = -1 * neg_acc_sug    