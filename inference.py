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

def inference(input_path, model_path='best_model.h5'):