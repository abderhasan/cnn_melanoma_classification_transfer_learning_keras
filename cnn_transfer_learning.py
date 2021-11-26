'''
This code enables you to train a Convolutional Neural Network (CNN) using a pretrained model (transfer learning)
Abder-Rahman Ali
abder@cs.stir.ac.uk
'''

from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
import matplotlib
# Force matplotlib to not use any Xwindows backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

#from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras import models
from keras import layers
#from keras import optimizers
import tensorflow
from tensorflow.keras.optimizers import Adam
import ssl
import os
import cv2
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# path to the training, validation, and testing directories
train_directory = '/home/ubuntu/cnn-transfer-learning/Data/train'
validation_directory = '/home/ubuntu/cnn-transfer-learning/Data/validation'
test_directory = '/home/ubuntu/cnn-transfer-learning/Data/test'
results_directory = '/home/ubuntu/cnn-transfer-learning/Data/results'
number_of_training_samples = 748
number_of_validation_samples = 108
number_of_test_samples = 510
batch_size = 4
correct_classification = 0
number_of_test_images = 0
labels = []
predictions = []

ssl._create_default_https_context = ssl._create_unverified_context

# get back the convolutional part of a VGG network trained on ImageNet
conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(512,512,3))
conv_base.summary() 

# preprocess the data
train_data = ImageDataGenerator()
validation_data = ImageDataGenerator()
test_data = ImageDataGenerator()

train_features = np.zeros(shape=(number_of_training_samples,16,16,512))
train_labels = np.zeros(shape=(number_of_training_samples))

train_generator = train_data.flow_from_directory(
    train_directory,
    target_size=(512,512),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True)

i = 0
for inputs_batch, labels_batch in train_generator:
    features_batch = conv_base.predict(inputs_batch)
    train_features[i*batch_size:(i+1)*batch_size] = features_batch
    train_labels[i*batch_size:(i+1)*batch_size] = labels_batch
    i += 1
    if i * batch_size >= number_of_training_samples:
        break

train_features = np.reshape(train_features, (number_of_training_samples,16*16*512))

validation_features = np.zeros(shape=(number_of_validation_samples,16,16,512))
validation_labels = np.zeros(shape=(number_of_validation_samples))

validation_generator = validation_data.flow_from_directory(
    validation_directory,
    target_size=(512,512),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)

i = 0
for inputs_batch, labels_batch in validation_generator:
    features_batch = conv_base.predict(inputs_batch)
    validation_features[i*batch_size:(i+1)*batch_size] = features_batch
    validation_labels[i*batch_size:(i+1)*batch_size] = labels_batch
    i += 1
    if i * batch_size >= number_of_validation_samples:
        break

validation_features = np.reshape(validation_features, (number_of_validation_samples,16*16*512))

test_generator = test_data.flow_from_directory(
	test_directory,
	target_size=(512,512),
	batch_size=batch_size,
	class_mode='binary',
	shuffle=False)

# define the Convolutional Neural Network (CNN) model
model = models.Sequential() 
model.add(layers.Dense(1024,activation='relu',input_dim=16*16*512))
model.add(layers.Dense(1,activation='sigmoid'))

# compile the model
model.compile(loss='binary_crossentropy',
	optimizer=tensorflow.keras.optimizers.Adam(lr=0.01),
	metrics=['acc'])

# fit the model to the data
history = model.fit(train_features,
	train_labels,
	epochs=30,
	batch_size=batch_size,
	validation_data=(validation_features,validation_labels))

# save the model
model.save('benign_and_melanoma_transfer_learning.h5')

# generate accuracy and loss curves for the training process (history of accuracy and loss)
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

number_of_epochs = range(1,len(acc)+1)

plt.plot(number_of_epochs, acc, 'r', label='Training accuracy')
plt.plot(number_of_epochs, val_acc, 'g', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('accuracy.png')

plt.close()

plt.plot(number_of_epochs, loss, 'r', label='Training loss')
plt.plot(number_of_epochs, val_loss, 'g', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('loss.png')

plt.close()

# evaluate the model

# predict classes
for root, dirs, files in os.walk(test_directory):
	for file in files:
		img = cv2.imread(root + '/' + file)
		img = cv2.resize(img,(512,512),interpolation=cv2.INTER_AREA) #<<<<<<< size
		img = np.expand_dims(img, axis=0)
		if os.path.basename(root) == 'nevus':
			label = 0
		elif os.path.basename(root) == 'melanoma':
			label = 1
		labels.append(label)
		feature_value = conv_base.predict(img)
		feature_value= np.reshape(feature_value,(1,16*16*512))
		img_class = model.predict(feature_value) 
		prediction = np.argmax(img_class,axis=1)
		predictions.append(prediction)
		if prediction == label:
			correct_classification = correct_classification + 1
		number_of_test_images = number_of_test_images + 1

print('number of correct results:')
print(correct_classification)

print('number of test images')
print(number_of_test_images)

print('Accuray:')
print(str((float(correct_classification)/float(number_of_test_images)) * 100) + '%')

#fpr, tpr, thresholds = roc_curve(labels, prediction_probabilities)
fpr, tpr, thresholds = roc_curve(labels, predictions)
auc_value = auc(fpr, tpr)

print('This is the AUC value')
print(auc_value)

# plot the ROC curve
plt.plot(fpr, tpr, label='CNN (area = {:.3f})'.format(auc_value))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.savefig('ROC.png')

plt.close()

# create the confusion matrix
cm = confusion_matrix(labels,predictions)
print(cm)