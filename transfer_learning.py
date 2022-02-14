# https://appliedmachinelearning.blog/2019/07/29/transfer-learning-using-feature-extraction-from-trained-models-food-images-classification/ 

PATH=os.path.expanduser('/Users/br/Documents/Semester 3 Beuth/learning from images/LFI project 2022/github synchro/LFI project 2022')


PATH_TRAINING="POTUS pics well ordered/training"
PATH_VALIDATION="POTUS pics well ordered/validation"
PATH_EVALUATION="POTUS pics well ordered/evaluation"

num_classes = 3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#print(PATH)
 
# Importing sklearn libraries
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, multilabel_confusion_matrix
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report


# Importing hypopt library for grid search
from hypopt import GridSearch
 
# Importing Keras libraries
#import keras
from keras.utils import np_utils
from keras.models import Sequential
#from keras.applications import VGG16
from tensorflow.keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D
 
import warnings
warnings.filterwarnings('ignore')

import os
import sys

os.chdir(PATH)

train = [os.path.join(PATH_TRAINING,img) for img in os.listdir(PATH_TRAINING)]
val = [os.path.join(PATH_VALIDATION,img) for img in os.listdir(PATH_VALIDATION)]
test = [os.path.join(PATH_EVALUATION,img) for img in os.listdir(PATH_EVALUATION)]

#len(train),len(val),len(test)
#train[0:5]

train_y = [int(img.split("/")[-1].split("_")[0]) for img in train]
val_y = [int(img.split("/")[-1].split("_")[0]) for img in val]
test_y = [int(img.split("/")[-1].split("_")[0]) for img in test]

# read
# Labelvergabe
# test, eval, train
# shuffle

 
# Convert class labels in one hot encoded vector
y_train = np_utils.to_categorical(train_y, num_classes)
y_val = np_utils.to_categorical(val_y, num_classes)
y_test = np_utils.to_categorical(test_y, num_classes)

train_y[0:10]
y_train[0:10]

print("Validation data available in " + str(num_classes)  + "classes")
[val_y.count(i) for i in range(0,num_classes)]

print("Test data available in "  + str(num_classes)  +  " classes")
[test_y.count(i) for i in range(0,num_classes)]



def show_imgs(X):
    plt.figure(figsize=(8, 8))
    k = 0
    for i in range(0,4):
        for j in range(0,4):
            image = load_img(train[k], target_size=(224, 224))
            plt.subplot2grid((4,4),(i,j))
            plt.imshow(image)
            k = k+1
    # show the plot
    plt.show()

# slows me down now
#show_imgs(train)

# chop the top dense layers, include_top=False
model = VGG16(weights="imagenet", include_top=False)
model.summary()
# output has dimensions 1, 7, 7, 512
input_size = 224
flatten_size = 7*7*512
model_name = "VGG16"

from tensorflow.keras.applications.inception_v3 import InceptionV3
model = InceptionV3(input_shape = (150, 150, 3), include_top = False, weights = 'imagenet')
model.summary()
pre_model = model
# output has dimensions 1, 3, 3, 2048
input_size = 150
flatten_size = 3*3*2048
model_name = "INCEPTIONV3"

from tensorflow.keras.applications import ResNet50
model = ResNet50(input_shape=(224, 224,3), include_top=False, weights="imagenet")
pre_model = model
# output has dimensions 1, 7, 7, 2048
input_size = 224
flatten_size = 7*7*2048
model_name = "RESNET50"


import tensorflow
model = VGG16(weights="imagenet", include_top=False)
model.summary()
model = tensorflow.keras.models.Model(inputs=model.input, outputs=model.get_layer('block4_pool').output)
model.summary()
pre_model = model
input_size = 224 
flatten_size = 14*14*512
model_name ="VGG16block4pool"
dataset= train


def create_features(dataset, pre_model):
 
    x_scratch = []
 
    # loop over the images
    for imagePath in dataset:
 
        # load the input image and image is resized to 224x224 (vgg16,resnet) or 150x150 pixels (inceptionv3)
        image = load_img(imagePath, target_size=(input_size, input_size))

        image = img_to_array(image)
 
        # preprocess the image by (1) expanding the dimensions and
        # (2) subtracting the mean RGB pixel intensity from the
        # ImageNet dataset
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)
 
        # add the image to the batch
        x_scratch.append(image)
 
    x = np.vstack(x_scratch)
    features = pre_model.predict(x, batch_size=32)
    features_flatten = features.reshape((features.shape[0], flatten_size))
    return x, features, features_flatten


# these lines take 1 hour
train_x, train_features, train_features_flatten = create_features(train, model)
val_x, val_features, val_features_flatten = create_features(val, model)
test_x, test_features, test_features_flatten = create_features(test, model)
 
print(train_x.shape, train_features.shape, train_features_flatten.shape)
print(val_x.shape, val_features.shape, val_features_flatten.shape)
print(test_x.shape, test_features.shape, test_features_flatten.shape)

# save the features in a single file
save_name = 'features'+model_name
np.savez(save_name, train_x=train_x, train_features=train_features, train_features_flatten=train_features_flatten,val_x=val_x, val_features=val_features, val_features_flatten=val_features_flatten,test_x=test_x, test_features=test_features, test_features_flatten=test_features_flatten)


# load features. Choose!

model_name = "VGG16"
model_name = "INCEPTIONV3"
model_name = "RESNET50"
model_name = "VGG16block4pool"

data= np.load('features'+model_name+'.npz')
train_x=data['train_x'] 
train_features=data['train_features']
train_features_flatten=data['train_features_flatten']
val_x=data['val_x']
val_features=data['val_features']
val_features_flatten=data['val_features_flatten']
test_x=data['test_x']
test_features=data['test_features']
test_features_flatten=data['test_features_flatten']

train_features_flatten.shape
test_features_flatten.shape


# Creating a checkpointer
checkpointer = ModelCheckpoint(filepath='scratchmodel.best.hdf5', monitor='val_accuracy', verbose=1,save_best_only=True)
other_checkpoint = ModelCheckpoint(model_name+'-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.h5', verbose=1, monitor='val_accuracy',save_best_only=True, mode='auto')  



def plot_acc_loss(history):
    fig = plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
 
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.suptitle(model_name+':')
    plt.show()

#plot_acc_loss(history)
# plt.show gives all kinds of weird results, empty plots etc --- instead mainly usage of save_acc_loss


def save_acc_loss(history,filename, type_of_network_string):
    fig = plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
 
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    #plt.show()
    plt.suptitle(model_name+': '+ type_of_network_string)
    plt.savefig(filename)

 
# Building up a Sequential model
model_scratch = Sequential()
model_scratch.add(Conv2D(32, (3, 3), activation='relu',input_shape = train_x.shape[1:]))
model_scratch.add(MaxPooling2D(pool_size=(2, 2)))
 
model_scratch.add(Conv2D(64, (3, 3), activation='relu'))
model_scratch.add(MaxPooling2D(pool_size=(2, 2)))
 
model_scratch.add(GlobalAveragePooling2D())
##
model_scratch.add(Dropout(0.2))
model_scratch.add(Dense(32, activation='relu'))
model_scratch.add(Dense(num_classes, activation='softmax'))
model_scratch.summary()

model_scratch.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model_scratch.summary()
 


#Fitting the model on the train data and labels.
history = model_scratch.fit(train_x, y_train,
          batch_size=32, epochs=20,
          verbose=1, callbacks=[checkpointer, other_checkpoint],
          validation_data=(val_x, y_val), shuffle=True)

'''
filepath='scratchmodel.best.hdf5'
#import keras
model = keras.models.load_model(filepath)
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.predict(test_features, batch_size=32)
model.evaluate(test_x, y_test)
model.evaluate(val_x, y_val)
'''

# save plot in file
type_of_network='Convolutional Neural Network'
save_acc_loss(history,'results/'+model_name+'sequentialnn'+'.png', type_of_network)


preds = np.argmax(model_scratch.predict(test_x), axis=1)
print("\nAccuracy on Test Data: ", accuracy_score(test_y, preds))
print("\nNumber of correctly identified imgaes: ",
      accuracy_score(test_y, preds, normalize=False),"\n")
confusion_matrix(test_y, preds, labels=range(0,num_classes))




model_transfer = Sequential()
model_transfer.add(GlobalAveragePooling2D(input_shape=train_features.shape[1:]))
model_transfer.add(Dropout(0.2))
model_transfer.add(Dense(100, activation='relu'))
model_transfer.add(Dense(num_classes, activation='softmax'))
model_transfer.summary()


model_transfer.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
history = model_transfer.fit(train_features, y_train, batch_size=32, epochs=250,
          validation_data=(val_features, y_val), callbacks=[checkpointer ],
          verbose=1, shuffle=True)

'''
#copy model and try it out. 
filepath='scratchmodel.best.hdf5'
#import keras
#model = keras.models.load_model(filepath)
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.predict(test_features, batch_size=32)
model.evaluate(test_features, y_test)
model.evaluate(val_features, y_val)
'''
#model_transfer.evaluate(test_features, y_test)


'''
model_transfer.evaluate(test_features, y_test)
test_features.shape
y_test.shape
train_features.shape
y_train.shape
'''


#plot_acc_loss(history)
# plt.show gives all kinds of weird results, empty plots etc --- don't use it, use save_acc_loss

# save png in file
type_of_network='Fully Connected Neural Network with Dropout'
save_acc_loss(history, 'results/'+model_name+'secondsequentialnn'+'.png', type_of_network)



'''
# new Network try: 
model_transfer = Sequential()
model_transfer.add(GlobalAveragePooling2D(input_shape=train_features.shape[1:]))
model_transfer.add(Dropout(0.5))
model_transfer.add(Dense(100, activation='relu'))
model_transfer.add(Dropout(0.5))
model_transfer.add(Dense(50, activation='relu'))
model_transfer.add(Dropout(0.5))
model_transfer.add(Dense(25, activation='relu'))
model_transfer.add(Dense(num_classes, activation='softmax'))
model_transfer.summary()


model_transfer.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
history = model_transfer.fit(train_features, y_train, batch_size=32, epochs=100,
          validation_data=(val_features, y_val), callbacks=[checkpointer],
          verbose=1, shuffle=True)


# save png in file
type_of_network='Fully Connected Neural Network with Dropout, 3 Layers'
save_acc_loss(history, 'results/'+model_name+'thirdsequentialnn'+'.png', type_of_network)
'''


# print out results of transfer-based nn
preds = np.argmax(model_transfer.predict(test_features), axis=1)
print("\nAccuracy on Test Data: ", accuracy_score(test_y, preds))
print("\nNumber of correctly identified imgaes: ",
      accuracy_score(test_y, preds, normalize=False),"\n")
confusion_matrix(test_y, preds, labels=range(0,num_classes))





logisticRegr = LogisticRegression(C=2, penalty="l2",solver="newton-cg",class_weight='balanced', multi_class="auto", max_iter=200, random_state=1)

train_score_vec = []
test_score_vec = []
parameters = 5*np.linspace(1,2000, num=20)/10**8
#parameters = np.linspace(9.9,10, num=10)/10

for C_var in parameters:
    logisticRegr = LogisticRegression(C=C_var, penalty="l2",solver="lbfgs")
    logisticRegr.fit(train_features_flatten, train_y)

    score = logisticRegr.score(train_features_flatten, train_y)
    print("train score")
    print(score)

    score1 = logisticRegr.score(test_features_flatten, test_y)
    print("test score")
    print(score1)
    train_score_vec.append(score)
    test_score_vec.append(score1)


fig = plt.figure(figsize=(5,5))
plt.plot(parameters, train_score_vec)
plt.plot(parameters, test_score_vec)
plt.title(model_name+': model accuracy of logistic regression')
plt.ylabel('accuracy')
plt.xlabel('penalization parameter C')
plt.legend(['train', 'validation'], loc='upper left')

# if you run first plt.show, then plt.savefig then the save will somehow be an empty picture
#in that case you have to run the whole plotting code block again
plt.savefig('results/'+model_name+'logisticregression'+'.png')
#plt.show()
# plt.show gives all kinds of weird results, empty plots etc --- don't use it





# https://www.datatechnotes.com/2020/07/classification-example-with-linearsvm-in-python.html
lsvc=LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
          verbose=0)

lsvc.fit(train_features_flatten, train_y)
score = lsvc.score(train_features_flatten, train_y)
print("Score: ", score)
score = lsvc.score(test_features_flatten, test_y)
print("Score: ", score)

pred_y = lsvc.predict(test_features_flatten)
cm = confusion_matrix(test_y, pred_y)
print(cm)

cr = classification_report(test_y, pred_y)
print(cr)


train_score_vec = []
test_score_vec = []
parameters = np.exp(np.linspace(-40,-20)) # we space exponentially!
#parameters = np.exp(np.linspace(-40,-1)) # we space exponentially!

for C_var in parameters:
    lsvc = LinearSVC(C=C_var, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
          verbose=0)
    lsvc.fit(train_features_flatten, train_y)

    score = lsvc.score(train_features_flatten, train_y)
    print("train score")
    print(score)

    score1 = lsvc.score(test_features_flatten, test_y)
    print("test score")
    print(score1)
    train_score_vec.append(score)
    test_score_vec.append(score1)

test_score_vec
train_score_vec


fig = plt.figure(figsize=(5,5))
plt.plot(parameters, train_score_vec)
plt.plot(parameters, test_score_vec)
plt.title(model_name+':\n accuracy of linear support vector classification')
plt.ylabel('accuracy')
plt.xlabel('penalization parameter C')
plt.legend(['train', 'validation'], loc='upper left')

# if you run first pltshow, then plt.savefig then the save will be empty picture
#in that case you have to run the whole plotting code block again
plt.savefig('results/'+model_name+'lsvc'+'.png')
#plt.show()






clf=RandomForestClassifier(n_estimators=100)
#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(train_features_flatten,y_train)

y_pred=clf.predict(test_features_flatten)
cr = classification_report(y_test, y_pred)
print(cr)

cm = confusion_matrix(y_test, y_pred)
print(cm)

mlcm = multilabel_confusion_matrix(y_test, y_pred)
print(mlcm)

acc=accuracy_score(y_test, y_pred)
print(acc)
# ist erbärmlich
# liegt wahrscheinlich daran, dass einzelne features nicht wichtig sind; aber eine lineare kombination der features sehr wichtig. damit ist random forest nicht gut


# randomizedsearchcv just takes a long time, accuracy still terrible, no point in computing this but can be mentioned when presenting
'''
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(train_features_flatten, y_train)

# winning params
rf_random.best_params_
# best estimator
rf_random.best_estimator_

accuracy_score(rf_random.best_estimator_.predict(test_features_flatten),y_test)
# pretty bad
'''