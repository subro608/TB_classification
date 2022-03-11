from tensorflow.keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract,LeakyReLU,Add,Average,Lambda,MaxPool2D,Dropout,UpSampling2D,Concatenate,Multiply,GlobalAveragePooling2D,Dense,ZeroPadding2D,AveragePooling2D
from tensorflow.keras.layers import concatenate,Flatten,ConvLSTM2D,LayerNormalization,GlobalAveragePooling2D,DepthwiseConv2D,SeparableConv2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.applications import VGG19, DenseNet121, ResNet50
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler 
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import Sequential, Input
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
import tensorflow as tf


def vgg(input_, class_num, tdata, tlabels, vdata, vlabels):
    in_model = VGG19(input_shape=input_,include_top=False,weights='imagenet',classes = class_num)
    inputs = Input(shape=input_)
    x1 = in_model(inputs)
    print(x1.shape)
    flat = Flatten()(x1)
    dense_1 = Dense(4096,activation = 'relu')(flat)
    dense_2 = Dense(4096,activation = 'relu')(dense_1)
    prediction = Dense(class_num,activation = 'softmax')(dense_2)
    in_pred45 = Model(inputs = inputs,outputs = prediction)
    in_pred45.compile(optimizer = SGD(learning_rate=0.0002), loss=CategoricalCrossentropy(from_logits = False) , metrics=['accuracy'])
    history = in_pred45.fit(tdata,tlabels,batch_size = 8,epochs = 1,validation_data = (vdata,vlabels))
    in_pred45.save_weights("vgg_weights.h5")
    return in_pred45, history

def densenet(input_, class_num,tdata, tlabels, vdata, vlabels):
    in_model2 = DenseNet121(input_shape=input_,include_top=False,weights='imagenet',classes = class_num)
    inputs = Input(shape=input_)
    x1 = in_model2(inputs)
    print(x1.shape)
    flat = Flatten()(x1)
    dense_1 = Dense(4096,activation = 'relu')(flat)
    dense_2 = Dense(4096,activation = 'relu')(dense_1)
    prediction = Dense(class_num,activation = 'softmax')(dense_2)
    in_pred = Model(inputs = inputs,outputs = prediction)  
    in_pred.compile(optimizer = SGD(learning_rate=0.0002), loss=CategoricalCrossentropy(from_logits = False) , metrics=['accuracy'])
    history = in_pred.fit(tdata,tlabels,batch_size = 8,epochs = 1,validation_data = (vdata,vlabels))
    in_pred.save_weights("densenet_weights.h5")
    return in_pred, history

def resnet(input_, class_num,tdata, tlabels, vdata, vlabels):
    in_model2 = ResNet50(input_shape=input_,include_top=False,weights='imagenet',classes = class_num)
    inputs = Input(shape=input_)
    x1 = in_model2(inputs)
    print(x1.shape)
    flat = Flatten()(x1)
    dense_1 = Dense(4096,activation = 'relu')(flat)
    dense_2 = Dense(4096,activation = 'relu')(dense_1)
    prediction = Dense(class_num,activation = 'softmax')(dense_2)
    in_pred4 = Model(inputs = inputs,outputs = prediction)
    in_pred4.compile(optimizer = SGD(learning_rate=0.0002), loss=CategoricalCrossentropy(from_logits = False) , metrics=['accuracy'])
    history = in_pred4.fit(tdata,tlabels,batch_size = 8,epochs = 1,validation_data = (vdata,vlabels))
    in_pred4.save_weights("resnet_weights.h5")
    return in_pred4, history
