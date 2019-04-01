from keras.datasets import fashion_mnist
import keras
from keras.models import load_model
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing import image
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#%matplotlib inline
import cv2


img_path_fist = "Fist/fist"
img_path_down = "Down/down"
img_path_up = "Up/up"
img_path_left = "Left/left"
img_path_right = "Right/right"
img_path_none = "None/none"
img_path_click1 = "Click1/click1"
img_path_click2 = "Click2/click2"

im = image.load_img(path="Fist/fist1.png",grayscale=True,target_size=(150,150,1))
im = image.img_to_array(im)
a = np.array([im])

for i in range(2,1500):
    im = image.load_img(path=img_path_fist + str(i) + ".png", grayscale = True, target_size=(150,150,1))
    im = image.img_to_array(im)
    b = np.array([im])
    a = np.concatenate((a,b), axis=0)
print("1")
for i in range(1,1500):
    im = image.load_img(path=img_path_up + str(i) + ".png", grayscale = True, target_size=(150,150,1))
    im = image.img_to_array(im)
    b = np.array([im])
    a = np.concatenate((a,b), axis=0)
print("2")
for i in range(1,1500):
    im = image.load_img(path=img_path_down + str(i) + ".png", grayscale = True, target_size=(150,150,1))
    im = image.img_to_array(im)
    b = np.array([im])
    a = np.concatenate((a,b), axis=0)
print("3")
for i in range(1,1500):
    im = image.load_img(path=img_path_left + str(i) + ".png", grayscale = True, target_size=(150,150,1))
    im = image.img_to_array(im)
    b = np.array([im])
    a = np.concatenate((a,b), axis=0)
print("4")
for i in range(1,1500):
    im = image.load_img(path=img_path_right + str(i) + ".png", grayscale = True, target_size=(150,150,1))
    im = image.img_to_array(im)
    b = np.array([im])
    a = np.concatenate((a,b), axis=0)
print("5")
for i in range(1,1500):
    im = image.load_img(path=img_path_none + str(i) + ".png", grayscale = True, target_size=(150,150,1))
    im = image.img_to_array(im)
    b = np.array([im])
    a = np.concatenate((a,b), axis=0)
print("6")
for i in range(1,1500):
    im = image.load_img(path=img_path_click1 + str(i) + ".png", grayscale = True, target_size=(150,150,1))
    im = image.img_to_array(im)
    b = np.array([im])
    a = np.concatenate((a,b), axis=0)
print("7")
for i in range(1,1500):
    im = image.load_img(path=img_path_click2 + str(i) + ".png", grayscale = True, target_size=(150,150,1))
    im = image.img_to_array(im)
    b = np.array([im])
    a = np.concatenate((a,b), axis=0)
print("8")
f = open('file4.txt', 'w+')
z = np.array([1 for i in range(1,1500)], np.int32)
b = np.array([2 for i in range(1,1500)], np.int32)
c = np.array([3 for i in range(1,1500)], np.int32)
d = np.array([4 for i in range(1,1500)], np.int32)
aa = np.array([5 for i in range(1,1500)], np.int32)
ab = np.array([6 for i in range(1,1500)], np.int32)
ac = np.array([7 for i in range(1,1500)], np.int32)
ad = np.array([8 for i in range(1,1500)], np.int32)
for i in range(len(z)):
    f.write("%i\n" % (z[i]))
for i in range(len(b)):
    f.write("%i\n" % (b[i]))
for i in range(len(c)):
    f.write("%i\n" % (c[i]))
for i in range(len(d)):
    f.write("%i\n" % (d[i]))
for i in range(len(aa)):
    f.write("%i\n" % (aa[i]))
for i in range(len(ab)):
    f.write("%i\n" % (ab[i]))
for i in range(len(ac)):
    f.write("%i\n" % (ac[i]))
for i in range(len(ad)):
    f.write("%i\n" % (ad[i]))
f.close()

data = np.loadtxt('file4.txt', delimiter=",")

classes = np.unique(data)
nClasses = len(classes)

a = a.reshape(-1, 150,150, 1)

a = a.astype('float32')
a = a / 255.
data_one_hot = to_categorical(data)

train_X,valid_X,train_label,valid_label = train_test_split(a, data_one_hot, test_size=0.2, random_state=13)

batch_size = 64
epochs = 10
num_classes = 9

fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(150,150,1),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(Dense(num_classes, activation='softmax'))

fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

fashion_train = fashion_model.fit(train_X, train_label, 
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  verbose=1,
                                  validation_data=(valid_X, valid_label))

fashion_model.save('trained_model4.h5')
