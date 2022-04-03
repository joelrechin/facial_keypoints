import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras 
from sklearn.model_selection import train_test_split

#Function to show an image.
def image_show(row):
    plt.imshow(image[row].reshape(96,96),cmap='gray')
    plt.show

def show_im_plot():
    x_features = X.filter(regex='x$')
    y_features = X.filter(regex='y$')
    for i in range(5):
        plt.subplot(2,5,i+1)
        plt.imshow(X_data(i).reshape(96,96))
        image_x_feat = x_features.loc[i]
        image_y_feat = y_features.loc[i]
        plt.plot(image_x_feat, image_y_feat, 'ro')
        plt.subplot(2, 5, i + 6)
        plt.imshow(X_test(i).reshape(96, 96))
        image_x_feat = result.loc[i]
        image_y_feat = y_predict.loc[i]
        plt.plot(image_x_feat, image_y_feat, 'ro')

#Read in facial keypoints csv data and fill any missing values.
df = pd.read_csv("FacialKeypoints.csv")
df.fillna(method = 'ffill', inplace=True)

#PROCESS DATA FOR THE CNN MODEL

X = df
y = df.loc[:, df.columns != 'Image']

#Convert the image data from the dataset into a numpy array for the model.
image = []
for i in range(0,7049):
    img = X['Image'][i].split(' ')
    img = ['0' if x == '' else x for x in img]
    image.append(img)
X_data = np.array(image,dtype='float')
X_data = X_data.reshape(-1,96,96,)
#Normalize the X data.
for row in X_data:
    row/=255

image_show(1)

#Convert the target data into a numpy array for the model.
target = []
for i in range(0,7049):
    targ = y.iloc[i,:]
    target.append(targ)
y_target = np.array(target,dtype='float')

#Make dataset larger by mirroring the images.
x_mirror = []
y_mirror = []

add_images = 1000

for i in range(add_images):
   x_mirror.append(np.fliplr(X_data[i]))
   y_mirror.append(y_target[i])
   y_array = np.array(y_mirror)

X_data = np.vstack([X_data, x_mirror])
y_target = np.vstack([y_target,y_mirror])

#Create training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X_data,y_target, test_size=.20, shuffle=True)
X_train_CNN = X_train.reshape(-1,96,96,1)
X_test_CNN = X_test.reshape(-1,96,96,1)

#CREATE THE CNN MODEL

model = tf.keras.Sequential()
model.add(keras.layers.Input(shape=[96,96,1]))
model.add(keras.layers.Convolution2D(32, kernel_size=(3,3), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Convolution2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Convolution2D(128, kernel_size=(3,3), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(30, activation=None))

#COMPILE THE CNN MODEL
model.compile(loss='mse', optimizer='adam',metrics=['mae','accuracy'])

#FIT THE CNN MODEL
history = model.fit(X_train_CNN,y_train, epochs=1, batch_size=255)

#CHECK PREDICTION

result = model.predict(X_test_CNN, verbose=1)












