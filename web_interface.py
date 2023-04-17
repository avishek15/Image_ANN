import gradio as gr
import numpy as np
from scipy.special import softmax
import tensorflow as tf


class MyModel:
    def __init__(self, tf_model_path=None):
        # self.num_classes = num_classes
        if tf_model_path is None:
            self.model = tf.keras.models.load_model("model_weights/ANN_2_layer.h5")
        else:
            self.model = tf.keras.models.load_model(tf_model_path)

    def predict(self, img):
        # print(np.max(img), np.min(img))
        # img = 255 - img
        return self.model.predict(img.reshape((1, 28, 28)), verbose=0)[0]


model = MyModel()


def classify(image):
    if image is None:
        return {str(i): 0. for i in range(10)}
    test_pred = model.predict(image)
    print(np.argmax(test_pred))
    return {str(i): test_pred[i].astype(float) for i in range(10)}


sketchpad = gr.Sketchpad()
label = gr.Label(num_top_classes=3)
interface = gr.Interface(fn=classify,
                         inputs=sketchpad,
                         outputs=label,
                         live=True).launch(height=400, width=400)
