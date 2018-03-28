import csv 
import cv2

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
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurement)
	
X_train = np.array(images)
y_train = np.array(measurements)

# Preprocessing

# Crop
#X_train = X_Train[:,:,:,:]
# Resize


print('X_train.shape',X_train.shape)
print('y_train.shape',y_train.shape)


#MODEL ARCHITECTURE

from keras.models import Sequential
from keras.layers import Flatten,Dense,Activation

model = Sequential()
model.add(Flatten(input_shape = (160,320,3)))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True)

model.save('model.h5')








	