import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Flatten, Dense

lr = 0.001
epochs = 10
batch_size = 50


(x_train,y_train), (x_test,y_test) = mnist.load_data()

x_train_shaped = x_train.reshape(x_train.shape[0], 28,28,1)
x_test_shaped = x_test.reshape(x_test.shape[0],28,28,1)

x_train_shaped = x_train_shaped.astype('float32')
x_test_shaped = x_test_shaped.astype('float32')
x_train_shaped /= 255
x_test_shaped /= 255

y_train_categorical = keras.utils.to_categorical(y_train,10)
y_test_categorical = keras.utils.to_categorical(y_test,10)

model = Sequential()

model.add(Conv2D(32, kernel_size=(5,5),strides=(1,1),padding='SAME',activation='relu',input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(64, kernel_size=(5,5), strides=(1,1),padding='SAME',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics= ['accuracy'])


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

model.fit(x_train_shaped, y_train_categorical, batch_size=batch_size,epochs=epochs,validation_split = 0, validation_data= None,callbacks=[history])

print(history.acc)
print(model.evaluate(x_test_shaped,y_test_categorical))
