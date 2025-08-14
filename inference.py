import tensorflow as tf
import os
import numpy as np
from model.densenet import DenseNet
from data.preprocessing import Preprocessor

def load_best_model(model_path='best_model.h5'):
    model = DenseNet(num_classes=2)
    
    dummy_input = tf.random.normal((1, 224, 224, 3))
    _ = model(dummy_input)
    
    model.load_weights(model_path)
    
    return model

def preprocess(img_path, img_shape=(224,224)):
    preprocessor = Preprocessor(img_shape=img_shape, augment=False)
    
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.cast(img, tf.float32)



def inference(input_path, model_path='best_model.h5'):