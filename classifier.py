import tensorflow as tf
import numpy as np
import pickle
from random import shuffle


def load_data():
    with open("dataset/mnist/train_df.pkl", "rb") as f:
        df = pickle.load(f)
    images = df['images']
    labels = df['labels']
    # indexes = np.arange(len(images))
    # shuffle(indexes)
    train_img = images  # [indexes[:int(0.9 * len(indexes))]]
    train_lbl = labels  # [indexes[:int(0.9 * len(indexes))]]
    with open("dataset/mnist/test_df.pkl", "rb") as f:
        df = pickle.load(f)
    timages = df['images']
    tlabels = df['labels']
    test_img = timages  # [indexes[int(0.9 * len(indexes)):]]
    test_lbl = tlabels  # [indexes[int(0.9 * len(indexes)):]]

    return train_img, train_lbl, test_img, test_lbl


x_train, y_train, x_test, y_test = load_data()

x_train = x_train #/ 255.
x_test = x_test #/ 255.

model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='elu'),
    tf.keras.layers.Dense(128, activation='elu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='Nadam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

model.save('model_weights/ANN_2_layer')
model.save_weights('model_weights/ANN_2_layer_wts')
