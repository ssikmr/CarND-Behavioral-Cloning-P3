import csv 
import cv2
import tensorflow as tf
import keras.backend as ktf
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
LEFT_CORR = 0.1
RIGHT_CORR = 0.2
DROP_RATE = 0.50
	
for line in lines:
	centre_file = './data/IMG/'+ line[0].split('/')[-1]
	left_file   = './data/IMG/'+ line[1].split('/')[-1]
	right_file  = './data/IMG/'+ line[2].split('/')[-1]
	if np.random.uniform() < DROP_RATE:
		image = cv2.imread(centre_file )
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image =image[60:140,:,:]
		measurement = float(line[3])
		images.append(image)
		measurements.append(measurement)
	
	#image =cv2.resize(image, (64,64))
	#if measurement > -0.05 && measurement < 0.05:
	image = cv2.imread(left_file )
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image =image[60:140,:,:]
	measurement = float(line[3]) + LEFT_CORR
	images.append(image)
	measurements.append(measurement)
	
	image = cv2.imread(right_file )
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image =image[60:140,:,:]
	measurement = float(line[3]) - RIGHT_CORR
	images.append(image)
	measurements.append(measurement)
	
	
    
	
    
X_train = np.array(images)
y_train = np.array(measurements)

print('X_train.shape',X_train.shape)
print('y_train.shape',y_train.shape)
# Preprocessing

# Crop done in Drive.py


# Resize
def resize(X):
	#return ktf.resize_images(X,64.0/80, 64.0/320, "channels_first")
	import tensorflow as tf
	return tf.image.resize_images(X,[32,128])




#MODEL ARCHITECTURE

from keras.models import Sequential
from keras.layers import Flatten,Dense,Activation,Lambda,Convolution2D,Dropout

model = Sequential()
#model.add(Lambda(lambda x: x/127.5 - 1,input_shape = (64,64,3)))
model.add(Lambda(lambda x: x/127.5 - 1,input_shape = (80,320,3)))
model.add(Lambda(resize))

model.add(Convolution2D(6,3,3))
model.add(Activation('relu'))

model.add(Convolution2D(12,3,3))
model.add(Activation('relu'))

#model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(300))
model.add(Activation('relu'))
#model.add(Dropout(0.5))

model.add(Dense(100))
model.add(Activation('relu'))
#model.add(Dropout(0.5))

model.add(Dense(30))
model.add(Activation('relu'))

model.add(Dense(1))


model.compile(loss = 'mse', optimizer = 'adam')
history_object = model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 2, batch_size = 128)
### print the keys contained in the history object
print(history_object.history.keys())

# ### plot the training and validation loss for each epoch
# plt.plot(history_object.history['loss'])
# plt.plot(history_object.history['val_loss'])
# plt.title('model mean squared error loss')
# plt.ylabel('mean squared error loss')
# plt.xlabel('epoch')
# plt.legend(['training set', 'validation set'], loc='upper right')
# plt.show()

model.save('model.h5')

