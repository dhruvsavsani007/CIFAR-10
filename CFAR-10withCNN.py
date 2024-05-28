import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape)
# print(x_train[0].shape)

# plt.imshow(x_train[0])
# plt.imshow(x_train[12])
# plt.show()

# print(x_train[0])
# print(x_train[0].shape)
# print(x_train.max())

x_train = x_train/225
x_test = x_test/255
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_train[0])
y_cat_train = to_categorical(y_train,10)
# print(y_cat_train.shape)
# print(y_cat_train[0])
y_cat_test = to_categorical(y_test,10)


model = Sequential()
# FIRST SET OF LAYERS

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

# SECOND SET OF LAYERS

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

# FLATTEN IMAGES FROM 28 by 28 to 764 BEFORE FINAL LAYER
model.add(Flatten())

# 256 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)
model.add(Dense(256, activation='relu'))

# LAST LAYER IS THE CLASSIFIER, THUS 10 POSSIBLE CLASSES
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# print(model.summary())
early_stop = EarlyStopping(monitor='val_loss',patience=3)
hist = model.fit(x_train,y_cat_train,epochs=15,validation_data=(x_test,y_cat_test),callbacks=[early_stop])

model.save('CFAR-10withCNN.h5')

losses = pd.DataFrame(hist.history)
losses.to_csv('CFAR-10withCNN.csv', index=False)
# print(losses.head())
# losses[['accuracy','val_accuracy']].plot()
# plt.show()
# losses[['loss','val_loss']].plot()
# plt.show()

# print(model.metrics_names)
# print(model.evaluate(x_test,y_cat_test,verbose=0))
predictions = model.predict_classes(x_test)
# print(classification_report(y_test,predictions))
# print(confusion_matrix(y_test,predictions))
# sns.heatmap(confusion_matrix(y_test,predictions),annot=True)
# plt.show()

my_image = x_test[16]
# plt.imshow(my_image)
# plt.show()

print(model.predict_classes(my_image.reshape(1,32,32,3)))