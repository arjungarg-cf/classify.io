# -*- coding: utf-8 -*-
"""Transfer_learning_ML.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ROtoCRuTAZh8QAWnFoyYrRxBV1cBHfaY
"""

!pip3 install kaggle
!mkdir .kaggle

import json
token = {"username":"shreyansh17109","key":"c3caeca809422b71f8e539911aee1ede"}
with open('/content/.kaggle/kaggle.json', 'w') as file:
    json.dump(token, file)

!cp /content/.kaggle/kaggle.json ~/.kaggle/kaggle.json

!kaggle config set -n path -v{/content}

!chmod 600 /root/.kaggle/kaggle.json
!kaggle datasets list -s waste

!kaggle datasets download -d techsash/waste-classification-data -p /content

!unzip \*.zip

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

width, height = 256, 256
train_dir = "DATASET/TRAIN"
test_dir = "DATASET/TEST"
batch_size = 16
epochs = 10

model = applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(width, height, 3))

from keras import backend as K
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

for layer in model.layers[:]:
    layer.trainable = False
x = model.output
x = Flatten()(x)
x = Dense(100, activation='relu')(x)
x = Dense(2, activation='softmax')(x)

# final_svm = Dense(100, activation="softmax")(x)

# model_svm = Model(input = model.input, output = final_svm)
# model_svm.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0002, momentum=0.91), metrics=["accuracy"])

model_final = Model(input = model.input, output = x)
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0002, momentum=0.93), metrics= ['accuracy', 'mae','binary_accuracy',f1,recall,precision])

from keras.applications.inception_v3 import preprocess_input


train_generator = ImageDataGenerator(
  preprocessing_function=preprocess_input,featurewise_center=True,
    featurewise_std_normalization=True).flow_from_directory(
  train_dir,
  target_size = (width,height),
  batch_size = batch_size,
  class_mode = "categorical")

train_generator_svm = ImageDataGenerator(
  preprocessing_function=preprocess_input,featurewise_center=True,
    featurewise_std_normalization=True).flow_from_directory(
  train_dir,
  target_size = (width,height),
  batch_size = 1,
  class_mode = "binary")

validation_generator = ImageDataGenerator(
  preprocessing_function=preprocess_input,featurewise_center=True,
    featurewise_std_normalization=True).flow_from_directory(
  test_dir,
  target_size = (width,height),
  class_mode = "categorical")

checkpoint = ModelCheckpoint("resnet50_retrain.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

model_final.fit_generator(
  train_generator,
  steps_per_epoch = 50,
  epochs = epochs,
  validation_data = validation_generator,
  callbacks = [checkpoint, early])

# reference:https://gist.github.com/liudanking/a35d0eba1855ef114922bd30de0a958c

model_final.summary()

intermediate_layer_model = Model(inputs=model_final.input,
                                 outputs=model_final.get_layer('dense_6').output)

for inputs_batch, labels_batch in train_generator_svm:
    print(inputs_batch.shape, labels_batch.shape)
    break

from keras import backend as K

import numpy as np
X = []  
Y = []
i=0
num1 =0
num0 =0
for inputs_batch, labels_batch in train_generator_svm:
    features_batch = intermediate_layer_model.predict(inputs_batch)
    # print(labels_batch)
    # break
    if i == 0:
      X = np.array(features_batch)
      Y = np.array(labels_batch)
    elif i%7 == 0:
      if labels_batch[0] > 0:
        num1+=1
      else:
        num0+=1
      X = np.append(X,features_batch,axis=0)
      Y = np.append(Y,labels_batch,axis=0)
    i=i+1
    if i % 700 == 0:
      print(i,num1,num0)
    if i > 15000:
      break

from sklearn import svm

estimator=svm.SVC(kernel='rbf',max_iter=100)
svm_lin=estimator.fit(X, Y)

print(svm_lin.score(X, Y))

import tensorflow as tf
print(tf.__version__)

model_svm.fit_generator(
  train_generator,
  steps_per_epoch = 50,
  epochs = epochs,
  validation_data = validation_generator,
  callbacks = [checkpoint, early])