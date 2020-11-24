import tensorflow as tf
import numpy as np
from tensorflow import keras

net = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=5, strides=4, activation='relu'),
    keras.layers.MaxPool2D(pool_size=3, strides=2),
    keras.layers.Conv2D(filters=256, kernel_size=5, padding='same', activation='relu'),
    keras.layers.MaxPool2D(pool_size=3, strides=2),
    keras.layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'),
    keras.layers.Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'),
    keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
    keras.layers.MaxPool2D(pool_size=3, strides=2),
    keras.layers.Flatten(),
    keras.layers.Dense(120, activation='relu'),
    # keras.layers.Dropout(0.5),
    keras.layers.Dense(84, activation='relu'),
    # keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='sigmoid')
])


class DataLoader():
    def __init__(self):
        fashion_mnist = keras.datasets.fashion_mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = fashion_mnist.load_data()
        self.train_images = np.expand_dims(self.train_images.astype(np.float32) / 255.0, axis=-1)
        self.test_images = np.expand_dims(self.test_images.astype(np.float32) / 255.0, axis=-1)
        self.train_labels = self.train_labels.astype(np.int32)
        self.test_labels = self.test_labels.astype(np.int32)
        self.num_train, self.num_test = self.train_images.shape[0], self.test_images.shape[0]

    def get_bitch_train(self, batch_size):
        index = np.random.randint(0, np.shape(self.train_images)[0], batch_size)
        resized_images = tf.image.resize_with_pad(self.train_images[index], 224, 224)
        return resized_images.numpy(), self.train_labels[index]

    def get_bitch_test(self, batch_size):
        index = np.random.ranint(0, np.shape(self.test_images)[0], batch_size)
        resized_images = tf.image_resize_with_pad(self.test_images[index], 224, 224)
        return resized_images.numpy(), self.test_labels[index]


batch_size = 128
dataLoader = DataLoader()
x_batch, y_batch = dataLoader.get_bitch_train(batch_size)
print('X_batch shape', x_batch.shape, 'y_batch shape', y_batch.shape)


def train_alexnet():
    epochs = 1
    num_iter = int(dataLoader.num_train / batch_size)
    for e in range(epochs):
        for n in range(num_iter):
            x_batch, y_batch = dataLoader.get_bitch_train(batch_size)
            net.fit(x_batch, y_batch)
            if n % 20 == 0:
                net.save_weights('5.6_alexnet_weights.h5')


optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0, nesterov=False)
net.compile(optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
# x_batch,y_batch=dataLoader.get_bitch_train(batch_size)
#et.fit(x_batch,y_batch,epochs =0)

#net.load_weights('5.6_alexnet_weights.h5')
train_alexnet()