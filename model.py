import csv 
import cv2
import tensorflow as tf

import numpy as np

lines = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	first = True
	for line in reader:
		if first :
			first = False
			continue
		lines.append(line)
		
images = []
measurements = []
for line in lines:
	source_file = line[0]
	filename = source_file.split('/')[-1]
	current_path =  './data/IMG/'+ filename
	image = cv2.imread(current_path)
	image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	image =image[60:140,:,:]
	image =cv2.resize(image, (64,64))
    
	images.append(image)

	measurement = float(line[3])
	measurements.append(measurement)
    
X_train = np.array(images)
y_train = np.array(measurements)

print('X_train.shape',X_train.shape)
print('y_train.shape',y_train.shape)
# Preprocessing

# Crop done in Drive.py

# Resize
#def resize(x):
#	return tf.image.resize_images(x,[64,64])
#X_train = tf.image.resize_images(X_train, [64,64])






#MODEL ARCHITECTURE

from keras.models import Sequential
from keras.layers import Flatten,Dense,Activation,Lambda,Convolution2D,Dropout

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1,input_shape = (64,64,3)))

model.add(Convolution2D(6,5,5))
model.add(Activation('relu'))

model.add(Convolution2D(12,5,5))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(300))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(30))
model.add(Activation('relu'))

model.add(Dense(1))


model.compile(loss = 'mse', optimizer = 'adam')
history_object = model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 5)
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')ss




	